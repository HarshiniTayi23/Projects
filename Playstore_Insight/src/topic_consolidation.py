import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
import re

class AdvancedTopicConsolidator:
    """
    Advanced consolidation engine for merging semantically similar topics
    Implements multiple similarity measures and intelligent merging strategies
    """
    
    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        print(f"Consolidator initialized with {similarity_threshold:.0%} similarity threshold")
        
        # Domain-specific taxonomy for Swiggy/food delivery
        self.domain_taxonomy = {
            'delivery': ['delivery', 'deliver', 'shipped', 'sent', 'arrived', 'courier', 'driver'],
            'time': ['late', 'delay', 'slow', 'fast', 'quick', 'time', 'waiting', 'minutes', 'hours'],
            'quality': ['quality', 'fresh', 'stale', 'cold', 'hot', 'taste', 'flavor', 'good', 'bad'],
            'behavior': ['rude', 'polite', 'behavior', 'attitude', 'manner', 'conduct', 'impolite'],
            'technical': ['app', 'crash', 'loading', 'bug', 'glitch', 'error', 'technical', 'system'],
            'payment': ['payment', 'pay', 'money', 'charged', 'billing', 'transaction', 'refund'],
            'order': ['order', 'booking', 'placed', 'confirmed', 'cancelled', 'cancel'],
            'food': ['food', 'meal', 'dish', 'cuisine', 'restaurant', 'menu', 'cooking'],
            'service': ['service', 'support', 'help', 'customer', 'care', 'assistance'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'fee', 'charge', 'discount']
        }
    
    def consolidate_topics_advanced(self, topics):
        """
        Advanced consolidation using multiple similarity measures
        """
        if not topics:
            return []
        
        print(f"Starting advanced consolidation for {len(topics)} topics")
        
        # Step 1: Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(topics)
        
        # Step 2: Find high-similarity pairs
        similar_pairs = self._find_similar_pairs(similarity_matrix, topics)
        
        # Step 3: Build consolidation groups
        consolidation_groups = self._build_consolidation_groups(similar_pairs)
        
        # Step 4: Merge topics within each group
        consolidated_topics = self._merge_topic_groups(consolidation_groups, topics)
        
        print(f" Consolidated from {len(topics)} to {len(consolidated_topics)} topics")
        
        return consolidated_topics
    
    def _build_similarity_matrix(self, topics):
        """Build comprehensive similarity matrix using multiple measures"""
        n_topics = len(topics)
        similarity_matrix = np.zeros((n_topics, n_topics))
        
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                similarity = self._calculate_comprehensive_similarity(topics[i], topics[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        return similarity_matrix
    
    def _calculate_comprehensive_similarity(self, topic1, topic2):
        """Calculate similarity using multiple sophisticated measures"""
        similarities = []
        weights = []
        
        # 1. Lexical similarity (string-based)
        lexical_sim = self._lexical_similarity(topic1['topic'], topic2['topic'])
        similarities.append(lexical_sim)
        weights.append(0.25)
        
        # 2. Semantic similarity (domain taxonomy)
        semantic_sim = self._semantic_similarity(topic1['topic'], topic2['topic'])
        similarities.append(semantic_sim)
        weights.append(0.25)
        
        # 3. Terms similarity (if available)
        terms_sim = self._terms_similarity(topic1, topic2)
        similarities.append(terms_sim)
        weights.append(0.20)
        
        # 4. Context similarity (method and confidence)
        context_sim = self._context_similarity(topic1, topic2)
        similarities.append(context_sim)
        weights.append(0.15)
        
        # 5. Vector similarity (TF-IDF based)
        vector_sim = self._vector_similarity(topic1, topic2)
        similarities.append(vector_sim)
        weights.append(0.15)
        
        # Weighted combination
        return sum(w * s for w, s in zip(weights, similarities))
    
    def _lexical_similarity(self, topic1_name, topic2_name):
        """Advanced lexical similarity using multiple string metrics"""
        # Normalize topic names
        name1 = self._normalize_topic_name(topic1_name)
        name2 = self._normalize_topic_name(topic2_name)
        
        # Jaccard similarity on words
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Edit distance similarity
        edit_sim = self._edit_distance_similarity(name1, name2)
        
        # Longest common subsequence
        lcs_sim = self._lcs_similarity(name1, name2)
        
        # Combine metrics
        return (jaccard * 0.5 + edit_sim * 0.3 + lcs_sim * 0.2)
    
    def _semantic_similarity(self, topic1_name, topic2_name):
        """Semantic similarity using domain taxonomy"""
        words1 = set(self._normalize_topic_name(topic1_name).split())
        words2 = set(self._normalize_topic_name(topic2_name).split())
        
        # Find which taxonomy categories each topic belongs to
        categories1 = self._get_taxonomy_categories(words1)
        categories2 = self._get_taxonomy_categories(words2)
        
        if not categories1 or not categories2:
            return 0.0
        
        # Calculate category overlap
        common_categories = categories1.intersection(categories2)
        total_categories = categories1.union(categories2)
        
        return len(common_categories) / len(total_categories) if total_categories else 0.0
    
    def _terms_similarity(self, topic1, topic2):
        """Similarity based on topic terms/keywords"""
        terms1 = set(topic1.get('terms', []))
        terms2 = set(topic2.get('terms', []))
        
        if not terms1 or not terms2:
            return 0.0
        
        intersection = len(terms1.intersection(terms2))
        union = len(terms1.union(terms2))
        
        return intersection / union if union > 0 else 0.0
    
    def _context_similarity(self, topic1, topic2):
        """Similarity based on context (method, confidence, support)"""
        similarities = []
        
        # Method similarity
        method1 = topic1.get('method', '')
        method2 = topic2.get('method', '')
        method_sim = 1.0 if method1 == method2 else 0.0
        similarities.append(method_sim)
        
        # Confidence similarity
        conf1 = topic1.get('confidence', 0.5)
        conf2 = topic2.get('confidence', 0.5)
        conf_sim = 1.0 - abs(conf1 - conf2)
        similarities.append(conf_sim)
        
        # Support similarity  
        supp1 = topic1.get('support', 1)
        supp2 = topic2.get('support', 1)
        max_supp = max(supp1, supp2)
        supp_sim = min(supp1, supp2) / max_supp if max_supp > 0 else 1.0
        similarities.append(supp_sim)
        
        return np.mean(similarities)
    
    def _vector_similarity(self, topic1, topic2):
        """Vector similarity using TF-IDF representation"""
        try:
            # Create document representations
            doc1 = f"{topic1['topic']} {' '.join(topic1.get('terms', []))}"
            doc2 = f"{topic2['topic']} {' '.join(topic2.get('terms', []))}"
            
            # Vectorize
            vectors = self.vectorizer.fit_transform([doc1, doc2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
            
        except Exception:
            return 0.0
    
    def _normalize_topic_name(self, topic_name):
        """Normalize topic name for better comparison"""
        # Convert underscores to spaces
        normalized = topic_name.replace('_', ' ')
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _get_taxonomy_categories(self, words):
        """Get taxonomy categories for a set of words"""
        categories = set()
        
        for word in words:
            for category, category_words in self.domain_taxonomy.items():
                if word in category_words:
                    categories.add(category)
        
        return categories
    
    def _edit_distance_similarity(self, str1, str2):
        """Calculate similarity based on edit distance"""
        if not str1 or not str2:
            return 0.0
        
        # Simple edit distance calculation
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        distance = self._levenshtein_distance(str1, str2)
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, str1, str2):
        """Calculate Levenshtein distance"""
        if len(str1) < len(str2):
            return self._levenshtein_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _lcs_similarity(self, str1, str2):
        """Calculate similarity based on longest common subsequence"""
        lcs_length = self._lcs_length(str1, str2)
        max_length = max(len(str1), len(str2))
        
        return lcs_length / max_length if max_length > 0 else 0.0
    
    def _lcs_length(self, str1, str2):
        """Calculate longest common subsequence length"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _find_similar_pairs(self, similarity_matrix, topics):
        """Find pairs of topics with similarity above threshold"""
        similar_pairs = []
        n_topics = len(topics)
        
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    similar_pairs.append((i, j, similarity_matrix[i][j]))
        
        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def _build_consolidation_groups(self, similar_pairs):
        """Build groups of topics that should be consolidated together"""
        # Union-Find data structure for grouping
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Group similar topics
        for i, j, similarity in similar_pairs:
            union(i, j)
        
        # Build groups
        groups = defaultdict(list)
        processed_indices = set()
        
        for i, j, similarity in similar_pairs:
            root_i = find(i)
            if root_i not in processed_indices:
                # Find all topics in this group
                group = []
                for idx in parent:
                    if find(idx) == root_i:
                        group.append(idx)
                groups[root_i] = group
                processed_indices.add(root_i)
        
        return list(groups.values())
    
    def _merge_topic_groups(self, consolidation_groups, topics):
        """Merge topics within each consolidation group"""
        consolidated_topics = []
        merged_indices = set()
        
        # Process each consolidation group
        for group in consolidation_groups:
            if len(group) > 1:
                # Merge topics in this group
                group_topics = [topics[i] for i in group]
                merged_topic = self._merge_multiple_topics(group_topics)
                consolidated_topics.append(merged_topic)
                
                # Mark these indices as merged
                merged_indices.update(group)
        
        # Add non-merged topics
        for i, topic in enumerate(topics):
            if i not in merged_indices:
                consolidated_topics.append(topic)
        
        return consolidated_topics
    
    def _merge_multiple_topics(self, topic_group):
        """Merge multiple similar topics into one consolidated topic"""
        # Select the best representative topic (highest confidence)
        best_topic = max(topic_group, key=lambda x: x.get('confidence', 0))
        
        # Combine metadata
        total_confidence = sum(t.get('confidence', 0) for t in topic_group) / len(topic_group)
        total_support = sum(t.get('support', 1) for t in topic_group)
        
        # Combine all terms
        all_terms = []
        for topic in topic_group:
            if 'terms' in topic:
                all_terms.extend(topic['terms'])
        
        # Get most frequent terms
        term_counts = Counter(all_terms)
        top_terms = [term for term, count in term_counts.most_common(10)]
        
        # Create merged topic
        merged_topic = {
            'topic': best_topic['topic'],
            'confidence': total_confidence,
            'method': 'advanced_consolidation',
            'support': total_support,
            'terms': top_terms,
            'merged_from': [t['topic'] for t in topic_group if t['topic'] != best_topic['topic']],
            'consolidation_score': len(topic_group)  # How many topics were merged
        }
        
        return merged_topic

if __name__ == "__main__":
    consolidator = AdvancedTopicConsolidator()
    print(" Advanced Topic Consolidation System initialized!")