"""
Graph-based transitive consistency checking and conflict resolution for entity matches.
"""

import logging
import networkx as nx
from typing import Dict, Set, List, Tuple, Any

logger = logging.getLogger(__name__)

def build_match_graph(results: Dict[str, Set[str]]) -> nx.Graph:
    """
    Build a bipartite graph where S1 entities are on one side, S2/S3 entities on the other,
    and edges represent predicted matches.
    
    Args:
        results: Dictionary mapping S1 entity IDs to a set of matching S2/S3 entity IDs.
        
    Returns:
        A NetworkX graph representing the matches.
    """
    graph = nx.Graph()
    for s1_id, s23_ids in results.items():
        graph.add_node(s1_id, bipartite=0)
        for s23_id in s23_ids:
            graph.add_node(s23_id, bipartite=1)
            graph.add_edge(s1_id, s23_id)
            
    return graph

def find_conflicts(graph: nx.Graph) -> List[Dict[str, Any]]:
    """
    Find S2/S3 nodes connected to multiple S1 entities (degree > 1 on the S1 side).
    
    Args:
        graph: Bipartite match graph.
        
    Returns:
        List of conflict dictionaries containing the conflicting S2/S3 entity and its matched S1 entities.
    """
    conflicts = []
    
    # Identify S2/S3 nodes (bipartite=1)
    s23_nodes = [n for n, d in graph.nodes(data=True) if d.get('bipartite') == 1]
    
    for s23_id in s23_nodes:
        # Get adjacent S1 nodes
        s1_neighbors = list(graph.neighbors(s23_id))
        if len(s1_neighbors) > 1:
            conflicts.append({
                's23_entity': s23_id,
                's1_entities': s1_neighbors,
                'description': f"S2/S3 entity {s23_id} matches multiple S1 entities: {s1_neighbors}"
            })
            
    return conflicts

def resolve_conflicts(results: Dict[str, Set[str]], match_scores: Dict[Tuple[str, str], float]) -> Dict[str, Set[str]]:
    """
    Resolve conflicts by keeping only the highest-confidence match for each S2/S3 entity.
    
    Args:
        results: Dictionary mapping S1 entity IDs to a set of matching S2/S3 entity IDs.
        match_scores: Dictionary mapping (s1_id, s23_id) tuples to match probabilities.
        
    Returns:
        Cleaned results dictionary.
    """
    if not results:
        return {}
        
    graph = build_match_graph(results)
    conflicts = find_conflicts(graph)
    
    if not conflicts:
        logger.info("No conflicts found during resolution.")
        return {k: set(v) for k, v in results.items()}
        
    # Create a deep copy of results to modify
    cleaned_results = {k: set(v) for k, v in results.items()}
    resolved_count = 0
    
    for conflict in conflicts:
        s23_id = conflict['s23_entity']
        s1_entities = conflict['s1_entities']
        
        # Find the best match
        best_s1 = None
        best_score = -1.0
        
        for s1_id in s1_entities:
            score = match_scores.get((s1_id, s23_id), 0.0)
            if score > best_score:
                best_score = score
                best_s1 = s1_id
                
        # Remove lower-confidence edges
        for s1_id in s1_entities:
            if s1_id != best_s1:
                if s23_id in cleaned_results.get(s1_id, set()):
                    cleaned_results[s1_id].remove(s23_id)
                    logger.info(f"Resolved conflict for {s23_id}: kept match with {best_s1} (score: {best_score:.4f}), removed match with {s1_id}")
                    resolved_count += 1
                    
    logger.info(f"Resolved {resolved_count} conflict edges.")
    return cleaned_results

def check_transitive_consistency(results: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
    """
    Check for transitive consistency issues in the match graph.
    Flags connected components that might suggest problematic transitive matches.
    
    Args:
        results: Dictionary mapping S1 entity IDs to a set of matching S2/S3 entity IDs.
        
    Returns:
        List of warning dictionaries.
    """
    warnings = []
    if not results:
        return warnings
        
    graph = build_match_graph(results)
    
    # Find connected components
    components = list(nx.connected_components(graph))
    
    for i, component in enumerate(components):
        s1_nodes = [n for n in component if graph.nodes[n].get('bipartite') == 0]
        s23_nodes = [n for n in component if graph.nodes[n].get('bipartite') == 1]
        
        # If a component has multiple S1 nodes and multiple S2/S3 nodes, it might indicate a complex transitive conflict
        if len(s1_nodes) > 1 and len(s23_nodes) > 1:
            warnings.append({
                'component_index': i,
                's1_nodes': s1_nodes,
                's23_nodes': s23_nodes,
                'description': f"Complex component found with {len(s1_nodes)} S1 entities and {len(s23_nodes)} S2/S3 entities."
            })
            
    if warnings:
        logger.info(f"Found {len(warnings)} transitive consistency warnings.")
        
    return warnings

def apply_graph_consistency(results: Dict[str, Set[str]], match_scores: Dict[Tuple[str, str], float]) -> Dict[str, Set[str]]:
    """
    Apply the full graph consistency pipeline: build graph, find conflicts, and resolve them.
    
    Args:
        results: Dictionary mapping S1 entity IDs to a set of matching S2/S3 entity IDs.
        match_scores: Dictionary mapping (s1_id, s23_id) tuples to match probabilities.
        
    Returns:
        Cleaned results dictionary.
    """
    logger.info("Applying graph consistency checks...")
    
    # Original stats
    total_original_matches = sum(len(matches) for matches in results.values())
    logger.info(f"Starting with {total_original_matches} total matches across {len(results)} S1 entities.")
    
    # 1. & 2. & 3. Build graph, find conflicts and resolve
    cleaned_results = resolve_conflicts(results, match_scores)
    
    # 4. Check for any remaining transitive issues (for logging/warnings only)
    check_transitive_consistency(cleaned_results)
    
    # Final stats
    total_cleaned_matches = sum(len(matches) for matches in cleaned_results.values())
    logger.info(f"Finished with {total_cleaned_matches} total matches.")
    logger.info(f"Removed {total_original_matches - total_cleaned_matches} conflicting matches.")
    
    return cleaned_results
