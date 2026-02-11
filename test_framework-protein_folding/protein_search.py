#!/usr/bin/env python3
"""
Protein Similarity Search using ANN algorithms and BLAST comparison.

This script compares proteins to a database using various ANN methods
(LSH, Hypercube, IVF-Flat, IVF-PQ, Neural LSH) and BLAST, then outputs
a comprehensive comparison report.

Usage:
    python protein_search.py -d protein_vectors.dat -q targets.fasta -o results.txt -method all
"""

import argparse
import subprocess
import os
import sys
import time
import struct
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from subprocess import CalledProcessError


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Algorithm parameters - tuned for protein embedding space (320 dims)."""
    num_nearest: int = 50
    radius: float = 50.0  # L2 radius for range search in embedding space
    using_range: str = 'false'  # 'true' or 'false'
    
    # LSH parameters (optimized for 320-dim protein embeddings)
    lsh_k: int = 6        # hash functions per table
    lsh_L: int = 8        # number of hash tables
    lsh_w: float = 1.0    # bucket width
    
    # Hypercube parameters
    hypercube_kproj: int = 12   # projection dimensions (2^12 = 4096 vertices)
    hypercube_w: float = 1.5    # window width
    hypercube_M: int = 5000     # max points to check
    hypercube_probes: int = 2   # vertices to probe
    
    # IVF-Flat parameters
    ivfflat_nlist: int = 1000   # number of clusters
    ivfflat_nprobe: int = 50    # clusters to search
    
    # IVF-PQ parameters
    ivfpq_nlist: int = 1000
    ivfpq_nprobe: int = 50
    ivfpq_m: int = 16           # subquantizers (320 / 16 = 20 dims each)
    ivfpq_nbits: int = 8
    
    # Neural LSH parameters
    neural_index: str = 'data/neural_lsh_index.pth'
    neural_T: int = 50          # partitions to probe (top_t)
    neural_knn: int = 10        # k for kNN graph
    neural_blocks: int = 400    # KaHIP partitions (num_partitions)
    neural_epochs: int = 15     # training epochs
    neural_hidden: str = '128,128'  # hidden layer dimensions
    
    # Paths (relative to script directory)
    cpp_binary: str = '../algorithms/lsh-hypercube-ivf/bin/search'
    neural_build: str = '../algorithms/neural_lsh/nlsh_build.py'
    neural_search: str = '../algorithms/neural_lsh/nlsh_search.py'
    
    seed: int = 42


config = Config()


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class Neighbor:
    """A single neighbor result."""
    id: int
    distance: float
    protein_id: str = ""  # UniProt accession if available
    blast_identity: float = 0.0  # BLAST identity percentage
    in_blast_top_n: bool = False


@dataclass 
class QueryResult:
    """Results for a single query protein."""
    query_id: int
    query_protein_id: str
    neighbors: List[Neighbor] = field(default_factory=list)
    time_seconds: float = 0.0
    

@dataclass
class MethodResult:
    """Results from one ANN method."""
    method_name: str
    queries: List[QueryResult] = field(default_factory=list)
    total_time: float = 0.0
    qps: float = 0.0
    recall_at_n: float = 0.0


@dataclass
class BlastHit:
    """A single BLAST hit."""
    query_id: str
    subject_id: str
    identity: float
    evalue: float
    bitscore: float


# ============================================================================
# Argument Parsing
# ============================================================================

def get_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Compare proteins to database using ANN algorithms and BLAST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python protein_search.py -d data/protein_vectors.dat -q data/targets.fasta -o output/results.txt -method all
  python protein_search.py -d data/protein_vectors.dat -q data/targets.fasta -o output/results.txt -method lsh
        """
    )
    p.add_argument('-d', required=True, type=str, metavar='EMBEDS',
                   help="Protein embeddings file (output of protein_embed.py)")
    p.add_argument('-q', required=True, type=str, metavar='FASTA',
                   help="Query proteins FASTA file")
    p.add_argument('-o', required=True, type=str, metavar='OUTPUT',
                   help="Output results file")
    p.add_argument('-method', required=True, 
                   choices=['all', 'lsh', 'hypercube', 'neural', 'ivfflat', 'ivfpq', 'blast'],
                   help="Search method to use")
    p.add_argument('-N', type=int, default=50,
                   help="Number of nearest neighbors (default: 50)")
    p.add_argument('--query-embeds', type=str, metavar='FILE',
                   help="Pre-computed query embeddings file (if not provided, will generate)")
    p.add_argument('--blast-db', type=str, default='data/swissprot_db',
                   help="BLAST database path (default: data/swissprot_db)")
    p.add_argument('--swissprot', type=str, default='data/swissprot.fasta',
                   help="Swiss-Prot FASTA file for BLAST database creation")
    p.add_argument('--skip-blast', action='store_true',
                   help="Skip BLAST comparison")
    p.add_argument('--verbose', '-v', action='store_true',
                   help="Verbose output")
    
    args = p.parse_args()
    config.num_nearest = args.N
    
    return {
        'embeds': args.d,
        'queries': args.q,
        'output': args.o,
        'method': args.method,
        'query_embeds': args.query_embeds,
        'blast_db': args.blast_db,
        'swissprot': args.swissprot,
        'skip_blast': args.skip_blast,
        'verbose': args.verbose,
    }


# ============================================================================
# Utility Functions
# ============================================================================

def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def resolve_path(relative_path: str) -> str:
    """Resolve a path relative to the script directory."""
    return str(get_script_dir() / relative_path)


def run_command(argv: List[str], description: str = "", verbose: bool = False) -> Optional[subprocess.CompletedProcess]:
    """Run a command and handle errors."""
    if verbose:
        print(f"  Running: {' '.join(argv)}")
    
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            timeout=None,
            text=True,
            check=True
        )
        return result
    except FileNotFoundError as e:
        print(f"Error: Command not found: {argv[0]}")
        print(f"  Make sure the binary/script exists and is executable.")
        return None
    except CalledProcessError as e:
        print(f"Error running {description or argv[0]}:")
        print(f"  Return code: {e.returncode}")
        if e.stdout:
            print(f"  Stdout: {e.stdout[:500]}")
        if e.stderr:
            print(f"  Stderr: {e.stderr[:500]}")
        return None


def parse_fasta_ids(fasta_file: str) -> List[str]:
    """Extract protein IDs from a FASTA file."""
    ids = []
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Parse FASTA header - format varies but often: >sp|P12345|NAME
                header = line[1:].strip()
                parts = header.split('|')
                if len(parts) >= 2:
                    ids.append(parts[1])  # UniProt accession
                else:
                    ids.append(header.split()[0])  # First word
    return ids


def count_vectors(dat_file: str) -> int:
    """Count vectors in a .dat file (SIFT format)."""
    count = 0
    with open(dat_file, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('<I', dim_bytes)[0]
            f.seek(dim * 4, 1)  # Skip the vector data
            count += 1
    return count


def load_protein_id_mapping(embeds_file: str, fasta_file: Optional[str] = None) -> Dict[int, str]:
    """Load mapping from vector index to protein ID.
    
    First tries to load from a JSON mapping file (created by protein_embed.py).
    Falls back to parsing the FASTA file if the mapping doesn't exist.
    """
    import json
    
    # Try JSON mapping file first (preferred)
    mapping_file = embeds_file.replace('.dat', '_ids.json')
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        # Convert string keys to int
        return {int(k): v for k, v in mapping.items()}
    
    # Fall back to FASTA file
    if fasta_file and os.path.exists(fasta_file):
        print(f"  Warning: No ID mapping file found ({mapping_file}), using FASTA file")
        mapping = {}
        ids = parse_fasta_ids(fasta_file)
        for i, protein_id in enumerate(ids):
            mapping[i] = protein_id
        return mapping
    
    print(f"  Warning: No ID mapping available")
    return {}


# ============================================================================
# Query Embedding Generation
# ============================================================================

def generate_query_embeddings(query_fasta: str, output_file: str, verbose: bool = False) -> bool:
    """Generate embeddings for query proteins using protein_embed_backend directly."""
    print("Generating query embeddings...")
    
    try:
        from protein_embed_backend import embed
        embed(
            input_file=query_fasta,
            output_file=output_file,
            batch_size=1,  # Small batch for queries
            n_first=0      # All query proteins
        )
        return True
    except Exception as e:
        print(f"Error generating query embeddings: {e}")
        return False


# ============================================================================
# C++ Algorithm Integration
# ============================================================================

def get_cpp_base_args(args: dict, output_file: str) -> List[str]:
    """Get base arguments for C++ search binary."""
    return [
        resolve_path(config.cpp_binary),
        '-d', args['embeds'],
        '-q', args['query_embeds'],
        '-o', output_file,
        '-N', str(config.num_nearest),
        '-type', 'sift',
        '-R', str(config.radius),
        '-range', config.using_range,
    ]


def run_lsh(args: dict, verbose: bool = False) -> Optional[MethodResult]:
    """Run Euclidean LSH algorithm."""
    print("Running LSH...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_file = f.name
    
    try:
        start_time = time.time()
        
        argv = get_cpp_base_args(args, output_file) + [
            '-lsh',
            '-k', str(config.lsh_k),
            '-L', str(config.lsh_L),
            '-w', str(config.lsh_w),
            '-seed', str(config.seed),
        ]
        
        result = run_command(argv, "LSH", verbose)
        if result is None:
            return None
        
        elapsed = time.time() - start_time
        method_result = parse_cpp_output(output_file, "Euclidean LSH", elapsed)
        return method_result
        
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def run_hypercube(args: dict, verbose: bool = False) -> Optional[MethodResult]:
    """Run Hypercube algorithm."""
    print("Running Hypercube...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_file = f.name
    
    try:
        start_time = time.time()
        
        argv = get_cpp_base_args(args, output_file) + [
            '-hypercube',
            '-kproj', str(config.hypercube_kproj),
            '-M', str(config.hypercube_M),
            '-probes', str(config.hypercube_probes),
            '-seed', str(config.seed),
        ]
        
        result = run_command(argv, "Hypercube", verbose)
        if result is None:
            return None
        
        elapsed = time.time() - start_time
        return parse_cpp_output(output_file, "Hypercube", elapsed)
        
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def run_ivfflat(args: dict, verbose: bool = False) -> Optional[MethodResult]:
    """Run IVF-Flat algorithm."""
    print("Running IVF-Flat...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_file = f.name
    
    try:
        start_time = time.time()
        
        argv = get_cpp_base_args(args, output_file) + [
            '-ivfflat',
            '-kclusters', str(config.ivfflat_nlist),
            '-nprobe', str(config.ivfflat_nprobe),
            '-seed', str(config.seed),
        ]
        
        result = run_command(argv, "IVF-Flat", verbose)
        if result is None:
            return None
        
        elapsed = time.time() - start_time
        return parse_cpp_output(output_file, "IVF-Flat", elapsed)
        
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def run_ivfpq(args: dict, verbose: bool = False) -> Optional[MethodResult]:
    """Run IVF-PQ algorithm."""
    print("Running IVF-PQ...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_file = f.name
    
    try:
        start_time = time.time()
        
        argv = get_cpp_base_args(args, output_file) + [
            '-ivfpq',
            '-kclusters', str(config.ivfpq_nlist),
            '-nprobe', str(config.ivfpq_nprobe),
            '-M', str(config.ivfpq_m),
            '-nbits', str(config.ivfpq_nbits),
            '-seed', str(config.seed),
        ]
        
        result = run_command(argv, "IVF-PQ", verbose)
        if result is None:
            return None
        
        elapsed = time.time() - start_time
        return parse_cpp_output(output_file, "IVF-PQ", elapsed)
        
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def parse_cpp_output(output_file: str, method_name: str, elapsed_time: float) -> MethodResult:
    """Parse the output file from C++ search binary."""
    result = MethodResult(method_name=method_name, total_time=elapsed_time)
    
    if not os.path.exists(output_file):
        return result
    
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Parse query results
    query_blocks = re.split(r'Query: (\d+)', content)
    
    for i in range(1, len(query_blocks), 2):
        query_id = int(query_blocks[i])
        block = query_blocks[i + 1] if i + 1 < len(query_blocks) else ""
        
        query_result = QueryResult(query_id=query_id, query_protein_id=f"query_{query_id}")
        
        # Parse neighbors
        neighbor_pattern = r'Nearest neighbor-(\d+): (\d+)\s+distanceApproximate: ([\d.]+)'
        for match in re.finditer(neighbor_pattern, block):
            rank = int(match.group(1))
            neighbor_id = int(match.group(2))
            distance = float(match.group(3))
            query_result.neighbors.append(Neighbor(id=neighbor_id, distance=distance))
        
        result.queries.append(query_result)
    
    # Parse summary metrics
    qps_match = re.search(r'QPS: ([\d.]+)', content)
    if qps_match:
        result.qps = float(qps_match.group(1))
    elif len(result.queries) > 0:
        result.qps = len(result.queries) / elapsed_time if elapsed_time > 0 else 0
    
    recall_match = re.search(r'Recall@N: ([\d.]+)', content)
    if recall_match:
        result.recall_at_n = float(recall_match.group(1))
    
    return result


# ============================================================================
# Neural LSH Integration
# ============================================================================

def run_neural_lsh(args: dict, verbose: bool = False) -> Optional[MethodResult]:
    """Run Neural LSH (build index if needed, then search)."""
    print("Running Neural LSH...")
    
    index_path = resolve_path(config.neural_index)
    
    # Build index if it doesn't exist
    if not os.path.exists(index_path):
        print("  Building Neural LSH index (first time only)...")
        success = build_neural_index(args, verbose)
        if not success:
            return None
    
    # Run search
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_file = f.name
    
    try:
        start_time = time.time()
        
        argv = [
            sys.executable, resolve_path(config.neural_search),
            '-d', args['embeds'],
            '-q', args['query_embeds'],
            '-i', index_path,
            '-o', output_file,
            '-type', 'sift',
            '-N', str(config.num_nearest),
            '-R', str(config.radius),
            '-T', str(config.neural_T),
            '--range', config.using_range,
        ]
        
        result = run_command(argv, "Neural LSH search", verbose)
        if result is None:
            return None
        
        elapsed = time.time() - start_time
        return parse_neural_output(output_file, elapsed)
        
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def build_neural_index(args: dict, verbose: bool = False) -> bool:
    """Build the Neural LSH index."""
    index_path = resolve_path(config.neural_index)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    argv = [
        sys.executable, resolve_path(config.neural_build),
        '-d', args['embeds'],
        '-i', index_path,
        '-type', 'sift',
        '--knn', str(config.neural_knn),
        '-m', str(config.neural_blocks),
        '--epochs', str(config.neural_epochs),
        '--seed', str(config.seed),
    ]
    
    result = run_command(argv, "Neural LSH build", verbose)
    return result is not None


def parse_neural_output(output_file: str, elapsed_time: float) -> MethodResult:
    """Parse output from Neural LSH search."""
    result = MethodResult(method_name="Neural LSH", total_time=elapsed_time)
    
    if not os.path.exists(output_file):
        return result
    
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Parse query results (similar format to C++ output)
    query_blocks = re.split(r'Query: (\d+)', content)
    
    for i in range(1, len(query_blocks), 2):
        query_id = int(query_blocks[i])
        block = query_blocks[i + 1] if i + 1 < len(query_blocks) else ""
        
        query_result = QueryResult(query_id=query_id, query_protein_id=f"query_{query_id}")
        
        neighbor_pattern = r'Nearest neighbor-(\d+): (\d+)\s+distanceApproximate: ([\d.]+)'
        for match in re.finditer(neighbor_pattern, block):
            neighbor_id = int(match.group(2))
            distance = float(match.group(3))
            query_result.neighbors.append(Neighbor(id=neighbor_id, distance=distance))
        
        result.queries.append(query_result)
    
    # Parse QPS
    qps_match = re.search(r'QPS: ([\d.]+)', content)
    if qps_match:
        result.qps = float(qps_match.group(1))
    elif len(result.queries) > 0:
        result.qps = len(result.queries) / elapsed_time if elapsed_time > 0 else 0
    
    recall_match = re.search(r'Recall@N: ([\d.]+)', content)
    if recall_match:
        result.recall_at_n = float(recall_match.group(1))
    
    return result


# ============================================================================
# BLAST Integration
# ============================================================================

def check_blast_installed() -> bool:
    """Check if BLAST+ is installed."""
    try:
        result = subprocess.run(['blastp', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_blast_database(fasta_file: str, db_path: str, verbose: bool = False) -> bool:
    """Create BLAST database from FASTA file."""
    if os.path.exists(db_path + '.phr') or os.path.exists(db_path + '.psq'):
        if verbose:
            print(f"  BLAST database already exists at {db_path}")
        return True
    
    print(f"  Creating BLAST database from {fasta_file}...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    argv = [
        'makeblastdb',
        '-in', fasta_file,
        '-dbtype', 'prot',
        '-out', db_path,
    ]
    
    result = run_command(argv, "BLAST database creation", verbose)
    return result is not None


def run_blast(query_fasta: str, db_path: str, num_hits: int = 50, 
              verbose: bool = False) -> Tuple[Optional[MethodResult], Dict[str, List[BlastHit]]]:
    """Run BLAST search and return results."""
    print("Running BLAST...")
    
    if not check_blast_installed():
        print("Error: BLAST+ is not installed. Install with: sudo apt install ncbi-blast+")
        return None, {}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        output_file = f.name
    
    try:
        start_time = time.time()
        
        # Run BLAST with tabular output format
        # Format: qseqid sseqid pident evalue bitscore
        argv = [
            'blastp',
            '-query', query_fasta,
            '-db', db_path,
            '-outfmt', '6 qseqid sseqid pident evalue bitscore',
            '-out', output_file,
            '-max_target_seqs', str(num_hits),
            '-num_threads', '4',
        ]
        
        result = run_command(argv, "BLAST search", verbose)
        if result is None:
            return None, {}
        
        elapsed = time.time() - start_time
        
        # Parse BLAST output
        blast_hits: Dict[str, List[BlastHit]] = {}  # query_id -> hits
        
        def extract_accession(full_id: str) -> str:
            """Extract accession number from sp|ACCESSION|NAME format."""
            if '|' in full_id:
                parts = full_id.split('|')
                if len(parts) >= 2:
                    return parts[1]
            return full_id.split()[0]
        
        with open(output_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    query_acc = extract_accession(parts[0])
                    subject_acc = extract_accession(parts[1])
                    hit = BlastHit(
                        query_id=query_acc,
                        subject_id=subject_acc,
                        identity=float(parts[2]),
                        evalue=float(parts[3]),
                        bitscore=float(parts[4]),
                    )
                    if hit.query_id not in blast_hits:
                        blast_hits[hit.query_id] = []
                    blast_hits[hit.query_id].append(hit)
        
        # Create MethodResult for comparison
        method_result = MethodResult(method_name="BLAST", total_time=elapsed)
        
        for query_id, hits in blast_hits.items():
            query_result = QueryResult(query_id=0, query_protein_id=query_id)
            for hit in hits[:num_hits]:
                query_result.neighbors.append(Neighbor(
                    id=0,  # We don't have vector IDs for BLAST
                    distance=100.0 - hit.identity,  # Convert identity to "distance"
                    protein_id=hit.subject_id,
                    blast_identity=hit.identity,
                ))
            method_result.queries.append(query_result)
        
        if len(blast_hits) > 0:
            method_result.qps = len(blast_hits) / elapsed if elapsed > 0 else 0
        
        return method_result, blast_hits
        
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


# ============================================================================
# Recall Calculation & Comparison
# ============================================================================

def calculate_recall_for_query(query_result: QueryResult, blast_hits: Dict[str, List[BlastHit]],
                                protein_mapping: Dict[int, str], n: int = 50) -> float:
    """Calculate Recall@N for a single query vs BLAST Top-N."""
    query_protein_id = query_result.query_protein_id
    if query_protein_id not in blast_hits:
        return 0.0
    
    blast_top_n = set(hit.subject_id for hit in blast_hits[query_protein_id][:n])
    
    # Get ANN top-N protein IDs
    ann_top_n = set()
    for neighbor in query_result.neighbors[:n]:
        if neighbor.id in protein_mapping:
            ann_top_n.add(protein_mapping[neighbor.id])
        elif neighbor.protein_id:
            ann_top_n.add(neighbor.protein_id)
    
    # Calculate overlap
    if len(blast_top_n) > 0:
        overlap = len(ann_top_n & blast_top_n)
        return overlap / len(blast_top_n)
    return 0.0


def calculate_recall_vs_blast(ann_result: MethodResult, blast_hits: Dict[str, List[BlastHit]],
                               protein_mapping: Dict[int, str], n: int = 50) -> float:
    """Calculate average Recall@N of ANN results vs BLAST Top-N across all queries."""
    if not blast_hits:
        return 0.0
    
    total_recall = 0.0
    query_count = 0
    
    for query_result in ann_result.queries:
        recall = calculate_recall_for_query(query_result, blast_hits, protein_mapping, n)
        if recall > 0 or query_result.query_protein_id in blast_hits:
            total_recall += recall
            query_count += 1
    
    return total_recall / query_count if query_count > 0 else 0.0


def annotate_with_blast(ann_result: MethodResult, blast_hits: Dict[str, List[BlastHit]],
                        protein_mapping: Dict[int, str]) -> None:
    """Add BLAST identity and annotation to ANN results."""
    for query_result in ann_result.queries:
        query_protein_id = query_result.query_protein_id
        
        # Get BLAST hits for this query (if any)
        query_blast_hits = blast_hits.get(query_protein_id, [])
        blast_lookup = {hit.subject_id: hit for hit in query_blast_hits}
        blast_top_n_ids = set(hit.subject_id for hit in query_blast_hits[:config.num_nearest])
        
        for neighbor in query_result.neighbors:
            # Always map the protein ID from index
            protein_id = protein_mapping.get(neighbor.id, "")
            neighbor.protein_id = protein_id
            
            # Add BLAST annotations if available
            if protein_id in blast_lookup:
                neighbor.blast_identity = blast_lookup[protein_id].identity
                neighbor.in_blast_top_n = protein_id in blast_top_n_ids
            else:
                neighbor.blast_identity = 0.0
                neighbor.in_blast_top_n = False


# ============================================================================
# Output Formatting
# ============================================================================

def write_results(output_file: str, all_results: List[MethodResult], 
                  blast_result: Optional[MethodResult], 
                  blast_hits: Dict[str, List[BlastHit]],
                  protein_mapping: Dict[int, str]) -> None:
    """Write results in the required PDF format."""
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Protein Similarity Search Results\n")
        f.write("=" * 80 + "\n\n")
        
        # Get query protein IDs
        query_ids = []
        if all_results and all_results[0].queries:
            query_ids = [q.query_protein_id for q in all_results[0].queries]
        
        for query_idx, query_protein_id in enumerate(query_ids):
            f.write(f"Query Protein: {query_protein_id}\n")
            f.write(f"N = {config.num_nearest} (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)\n\n")
            
            # [1] Summary comparison table - PDF format
            f.write("[1] Συνοπτική σύγκριση μεθόδων\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Method':<20} | {'Time/query (s)':<15} | {'QPS':<10} | {'Recall@N vs BLAST Top-N':<25}\n")
            f.write("-" * 80 + "\n")
            
            for result in all_results:
                time_per_query = result.total_time / len(result.queries) if result.queries else 0
                # Calculate recall for THIS specific query
                if query_idx < len(result.queries):
                    recall = calculate_recall_for_query(result.queries[query_idx], blast_hits, protein_mapping)
                else:
                    recall = 0.0
                f.write(f"{result.method_name:<20} | {time_per_query:<15.4f} | {result.qps:<10.1f} | {recall:<25.2f}\n")
            
            if blast_result:
                time_per_query = blast_result.total_time / len(blast_result.queries) if blast_result.queries else 0
                f.write(f"{'BLAST (Ref)':<20} | {time_per_query:<15.4f} | {blast_result.qps:<10.1f} | {'1.00 (ορίζει το Top-N)':<25}\n")
            
            f.write("-" * 80 + "\n\n")
            
            # [2] Detailed Top-N neighbors per method - PDF format
            f.write(f"[2] Top-N γείτονες ανά μέθοδο (εδώ π.χ. N = 10 για εκτύπωση)\n\n")
            
            display_n = min(10, config.num_nearest)  # Show top 10 for readability
            
            for result in all_results:
                f.write(f"Method: {result.method_name}\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Rank':<6} | {'Neighbor ID':<15} | {'L2 Dist':<10} | {'BLAST Identity':<14} | {'In BLAST Top-N?':<16} | {'Bio comment':<20}\n")
                f.write("-" * 100 + "\n")
                
                if query_idx < len(result.queries):
                    query_result = result.queries[query_idx]
                    for rank, neighbor in enumerate(query_result.neighbors[:display_n], 1):
                        in_blast = "Yes" if neighbor.in_blast_top_n else "No"
                        
                        # Generate bio comment per PDF requirements
                        bio_comment = ""
                        if neighbor.blast_identity > 0 and neighbor.blast_identity < 30:
                            if neighbor.in_blast_top_n:
                                bio_comment = "Remote homolog?"
                            else:
                                bio_comment = "Πιθανό false positive"
                        elif neighbor.blast_identity >= 30:
                            bio_comment = "--"
                        else:
                            bio_comment = "--"
                        
                        identity_str = f"{neighbor.blast_identity:.0f}%" if neighbor.blast_identity > 0 else "N/A"
                        
                        f.write(f"{rank:<6} | {neighbor.protein_id or str(neighbor.id):<15} | "
                               f"{neighbor.distance:<10.2f} | {identity_str:<14} | {in_blast:<16} | "
                               f"{bio_comment:<20}\n")
                
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        # Overall summary
        # Overall summary
        f.write("=" * 80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Method':<20} | {'Total Time (s)':<15} | {'Avg QPS':<12} | {'Recall@N vs BLAST':<20}\n")
        f.write("-" * 75 + "\n")
        
        for result in all_results:
            recall = calculate_recall_vs_blast(result, blast_hits, protein_mapping)
            f.write(f"{result.method_name:<20} | {result.total_time:<15.4f} | {result.qps:<12.1f} | {recall:<20.2f}\n")
        
        if blast_result:
            f.write(f"{'BLAST (Ref)':<20} | {blast_result.total_time:<15.4f} | {blast_result.qps:<12.1f} | {'1.00 (defines Top-N)':<20}\n")
        
        f.write("\n")
    
    print(f"\nResults written to: {output_file}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    args = get_args()
    verbose = args['verbose']
    
    print("=" * 60)
    print("Protein Similarity Search")
    print("=" * 60)
    
    # Validate inputs
    if not os.path.exists(args['embeds']):
        print(f"Error: Embeddings file not found: {args['embeds']}")
        print("Run protein_embed.py first to generate embeddings.")
        sys.exit(1)
    
    if not os.path.exists(args['queries']):
        print(f"Error: Query file not found: {args['queries']}")
        sys.exit(1)
    
    # Generate query embeddings if not provided
    if args['query_embeds'] and os.path.exists(args['query_embeds']):
        print(f"Using pre-computed query embeddings: {args['query_embeds']}")
    else:
        query_embed_file = args['query_embeds'] or 'data/query_vectors.dat'
        query_embed_file = resolve_path(query_embed_file)
        
        if not os.path.exists(query_embed_file):
            success = generate_query_embeddings(args['queries'], query_embed_file, verbose)
            if not success:
                print("Error: Failed to generate query embeddings")
                sys.exit(1)
        
        args['query_embeds'] = query_embed_file
    
    # Load protein ID mapping
    print("Loading protein ID mappings...")
    protein_mapping = load_protein_id_mapping(
        args['embeds'], 
        args.get('swissprot', '')
    )
    
    query_ids = parse_fasta_ids(args['queries'])
    
    # Run BLAST first (for comparison baseline - defines Top-N ground truth)
    blast_result = None
    blast_hits = {}
    
    if not args['skip_blast']:
        # Create BLAST database if needed
        if os.path.exists(args['swissprot']):
            create_blast_database(args['swissprot'], args['blast_db'], verbose)
            blast_result, blast_hits = run_blast(
                args['queries'], args['blast_db'], 
                config.num_nearest, verbose
            )
        else:
            print(f"Warning: Swiss-Prot file not found at {args['swissprot']}, skipping BLAST")
            print("         BLAST is required for Recall@N calculation.")
    
    # Run selected method(s)
    all_results: List[MethodResult] = []
    method = args['method']
    
    def process_result(result: MethodResult) -> MethodResult:
        """Update query IDs and annotate with BLAST."""
        # First update query protein IDs
        for i, q in enumerate(result.queries):
            if i < len(query_ids):
                q.query_protein_id = query_ids[i]
        # Then annotate with BLAST (needs correct query IDs)
        annotate_with_blast(result, blast_hits, protein_mapping)
        return result
    
    if method in ['all', 'lsh']:
        result = run_lsh(args, verbose)
        if result:
            all_results.append(process_result(result))
    
    if method in ['all', 'hypercube']:
        result = run_hypercube(args, verbose)
        if result:
            all_results.append(process_result(result))
    
    if method in ['all', 'ivfflat']:
        result = run_ivfflat(args, verbose)
        if result:
            all_results.append(process_result(result))
    
    if method in ['all', 'ivfpq']:
        result = run_ivfpq(args, verbose)
        if result:
            all_results.append(process_result(result))
    
    if method in ['all', 'neural']:
        result = run_neural_lsh(args, verbose)
        if result:
            all_results.append(process_result(result))
    
    # Write results
    if all_results or blast_result:
        write_results(args['output'], all_results, blast_result, blast_hits, protein_mapping)
        print("\nDone!")
    else:
        print("\nNo results to write. Check for errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
