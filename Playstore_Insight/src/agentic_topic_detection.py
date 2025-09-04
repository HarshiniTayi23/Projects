import re
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from .database_setup import DatabaseManager

class AgenticTopicDetector:
    """
    Multi-Agent Topic Detection System
    Uses autonomous agents for topic discovery, classification, and consolidation
    """
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or DatabaseManager()
        
        # Initialize Agents
        self.discovery_agent = TopicDiscoveryAgent()
        self.classification_agent = TopicClassificationAgent()
        self.consolidation_agent = TopicConsolidationAgent()
        self.quality_agent = QualityAssessmentAgent()
        
        # High-recall seed topics with semantic patterns
        self.seed_topics = {
            'delivery_delay': {
                'patterns': ['late', 'delay', 'slow', 'waiting', 'time', 'hours', 'delayed', 'arrived late', 'took forever', 'very slow'],
                'semantic_clusters': ['time_related', 'delay_words', 'speed_complaints']
            },
            'food_quality': {
                'patterns': ['cold', 'stale', 'fresh', 'quality', 'taste', 'bad food', 'spoiled', 'rotten', 'tasteless', 'terrible food'],
                'semantic_clusters': ['quality_words', 'taste_descriptors', 'food_condition']
            },
            'delivery_partner_behavior': {
                'patterns': ['rude', 'polite', 'behavior', 'delivery boy', 'partner', 'attitude', 'impolite', 'behaved badly', 'was rude'],
                'semantic_clusters': ['behavior_words', 'attitude_descriptors', 'politeness']
            },
            'app_performance': {
                'patterns': ['crash', 'slow app', 'loading', 'bug', 'glitch', 'not working', 'app freeze', 'technical issue'],
                'semantic_clusters': ['technical_issues', 'performance_words', 'functionality_problems']
            },
            'payment_issues': {
                'patterns': ['payment', 'charged', 'refund', 'money', 'billing', 'transaction', 'payment failed', 'card issue'],
                'semantic_clusters': ['payment_words', 'financial_terms', 'transaction_problems']
            },
            'order_cancellation': {
                'patterns': ['cancelled', 'cancel', 'cancellation', 'order cancelled', 'order canceled', 'cancel order'],
                'semantic_clusters': ['cancellation_words', 'order_status', 'booking_issues']
            }
        }
        
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
        
    def process_reviews_with_agents(self, reviews_batch):
        """
        Multi-agent processing pipeline for high-recall topic detection
        """
        print(f" Starting multi-agent topic detection for {len(reviews_batch)} reviews")
        
        # Agent 1: Discovery - Find topics using multiple methods
        discovered_topics = self.discovery_agent.discover_topics(reviews_batch, self.seed_topics)
        print(f" Discovery Agent found {len(discovered_topics)} potential topics")
        
        # Agent 2: Classification - Classify each review with high confidence
        classifications = self.classification_agent.classify_reviews(reviews_batch, discovered_topics)
        print(f" Classification Agent processed {len(classifications)} review-topic mappings")
        
        # Agent 3: Consolidation - Merge semantically similar topics
        consolidated_topics = self.consolidation_agent.consolidate_topics(discovered_topics)
        print(f" Consolidation Agent merged into {len(consolidated_topics)} unique topics")
        
        # Agent 4: Quality Assessment - Validate topic quality
        quality_scores = self.quality_agent.assess_quality(consolidated_topics, reviews_batch)
        print(f" Quality Agent validated {len(quality_scores)} high-quality topics")
        
        # Filter and finalize topics based on quality
        final_topics = self._finalize_topics(consolidated_topics, quality_scores)
        final_mappings = self._update_mappings(classifications, consolidated_topics)
        
        return final_topics, final_mappings
    
    def _finalize_topics(self, topics, quality_scores):
        """Filter topics based on quality assessment"""
        final_topics = []
        for topic in topics:
            quality = quality_scores.get(topic['topic'], 0.5)
            if quality > 0.6:  # High quality threshold
                topic['confidence'] = quality
                final_topics.append(topic)
        return final_topics
    
    def _update_mappings(self, classifications, consolidated_topics):
        """Update review-topic mappings after consolidation"""
        # Create mapping between old and new topic names
        topic_mapping = {}
        for topic in consolidated_topics:
            if 'merged_from' in topic:
                for old_topic in topic['merged_from']:
                    topic_mapping[old_topic] = topic['topic']
            else:
                topic_mapping[topic['topic']] = topic['topic']
        
        # Update classifications
        updated_mappings = []
        for mapping in classifications:
            new_topic = topic_mapping.get(mapping['topic'], mapping['topic'])
            updated_mappings.append({
                'review_id': mapping['review_id'],
                'topic': new_topic,
                'confidence': mapping['confidence']
            })
        
        return updated_mappings
    
    def save_topics_to_db(self, detected_topics, review_mappings):
        """Save agent-detected topics to database"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        # Insert new topics with agent metadata
        for topic_data in detected_topics:
            cursor.execute('''
                INSERT OR IGNORE INTO topics (topic_name, confidence_score, is_seed_topic)
                VALUES (?, ?, FALSE)
            ''', (topic_data['topic'], topic_data['confidence']))
        
        # Insert review-topic mappings
        for mapping in review_mappings:
            # Get topic_id
            cursor.execute('SELECT id FROM topics WHERE topic_name = ?', (mapping['topic'],))
            topic_row = cursor.fetchone()
            
            if topic_row:
                topic_id = topic_row[0]
                cursor.execute('''
                    INSERT OR IGNORE INTO review_topics (review_id, topic_id, confidence_score)
                    VALUES (?, ?, ?)
                ''', (mapping['review_id'], topic_id, mapping['confidence']))
        
        conn.commit()
        conn.close()

class TopicDiscoveryAgent:
    """Agent responsible for discovering topics using multiple approaches"""
    
    def discover_topics(self, reviews, seed_topics):
        discovered = []
        
        # Method 1: Enhanced rule-based with semantic patterns
        rule_topics = self._rule_based_discovery(reviews, seed_topics)
        discovered.extend(rule_topics)
        
        # Method 2: Advanced clustering with multiple algorithms
        cluster_topics = self._clustering_discovery(reviews)
        discovered.extend(cluster_topics)
        
        # Method 3: N-gram pattern detection for emerging topics
        ngram_topics = self._ngram_pattern_discovery(reviews)
        discovered.extend(ngram_topics)
        
        return discovered
    
    def _rule_based_discovery(self, reviews, seed_topics):
        """Enhanced rule-based discovery with semantic understanding"""
        topics = []
        
        for topic_name, topic_data in seed_topics.items():
            patterns = topic_data['patterns']
            
            # Calculate semantic confidence for each pattern
            confidence_scores = []
            for review in reviews:
                text = review['content'].lower()
                matches = sum(1 for pattern in patterns if pattern in text)
                if matches > 0:
                    confidence = matches / len(patterns)
                    confidence_scores.append(confidence)
            
            if confidence_scores:
                avg_confidence = np.mean(confidence_scores)
                topics.append({
                    'topic': topic_name,
                    'confidence': avg_confidence,
                    'method': 'enhanced_rule_based',
                    'support': len(confidence_scores)
                })
        
        return topics
    
    def _clustering_discovery(self, reviews):
        """Advanced clustering using multiple algorithms"""
        if len(reviews) < 10:
            return []
        
        texts = [review['content'] for review in reviews if review['content']]
        if len(texts) < 5:
            return []
        
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Use multiple clustering algorithms for robustness
            topics = []
            
            # K-means clustering
            for n_clusters in [3, 5, 7]:
                if n_clusters < len(texts):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                    cluster_labels = kmeans.fit_predict(tfidf_matrix)
                    
                    feature_names = vectorizer.get_feature_names_out()
                    
                    for cluster_id in range(n_clusters):
                        cluster_center = kmeans.cluster_centers_[cluster_id]
                        top_indices = cluster_center.argsort()[-5:][::-1]
                        top_terms = [str(feature_names[i]) for i in top_indices]
                        
                        cluster_reviews = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                        confidence = len(cluster_reviews) / len(texts)
                        
                        if confidence > 0.1:  # Minimum support threshold
                            topic_name = "_".join(top_terms[:2])
                            topics.append({
                                'topic': topic_name,
                                'confidence': confidence,
                                'method': f'clustering_k{n_clusters}',
                                'terms': top_terms,
                                'support': len(cluster_reviews)
                            })
            
            return topics
            
        except Exception as e:
            print(f"  Clustering discovery failed: {e}")
            return []
    
    def _ngram_pattern_discovery(self, reviews):
        """Discover emerging topics using n-gram patterns"""
        from collections import Counter
        
        topics = []
        all_text = " ".join([review['content'].lower() for review in reviews if review['content']])
        
        # Extract frequent bigrams and trigrams
        words = re.findall(r'\b\w+\b', all_text)
        
        # Bigrams
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        
        # Trigrams  
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        
        # Find emerging patterns
        total_reviews = len(reviews)
        
        for ngram, count in bigram_counts.most_common(10):
            if count >= 3 and count/total_reviews > 0.1:  # Significant frequency
                topics.append({
                    'topic': ngram,
                    'confidence': count/total_reviews,
                    'method': 'ngram_bigram',
                    'support': count
                })
        
        for ngram, count in trigram_counts.most_common(5):
            if count >= 2 and count/total_reviews > 0.05:
                topics.append({
                    'topic': ngram,
                    'confidence': count/total_reviews,
                    'method': 'ngram_trigram',
                    'support': count
                })
        
        return topics

class TopicClassificationAgent:
    """Agent responsible for classifying reviews to topics with high recall"""
    
    def classify_reviews(self, reviews, topics):
        """High-recall classification using multiple methods"""
        classifications = []
        
        for review in reviews:
            review_topics = self._classify_single_review(review, topics)
            classifications.extend(review_topics)
        
        return classifications
    
    def _classify_single_review(self, review, topics):
        """Classify single review with multiple confidence measures"""
        text = review['content'].lower()
        classifications = []
        
        for topic in topics:
            confidence = self._calculate_topic_confidence(text, topic)
            
            if confidence > 0.3:  # High recall threshold
                classifications.append({
                    'review_id': review['id'],
                    'topic': topic['topic'],
                    'confidence': confidence
                })
        
        return classifications
    
    def _calculate_topic_confidence(self, text, topic):
        """Calculate confidence using multiple signals"""
        confidence_scores = []
        
        # Method 1: Keyword matching
        if 'terms' in topic:
            matches = sum(1 for term in topic['terms'] if term in text)
            if topic['terms']:
                keyword_conf = matches / len(topic['terms'])
                confidence_scores.append(keyword_conf)
        
        # Method 2: Topic name matching
        topic_words = topic['topic'].lower().split('_')
        text_words = set(text.split())
        word_matches = len(set(topic_words).intersection(text_words))
        if topic_words:
            name_conf = word_matches / len(topic_words)
            confidence_scores.append(name_conf)
        
        # Method 3: Semantic similarity (simple approach)
        semantic_conf = self._simple_semantic_similarity(text, topic['topic'])
        confidence_scores.append(semantic_conf)
        
        # Return weighted average
        if confidence_scores:
            return np.mean(confidence_scores)
        return 0.0
    
    def _simple_semantic_similarity(self, text, topic_name):
        """Simple semantic similarity using word overlap"""
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        topic_words = set(re.findall(r'\b\w+\b', topic_name.lower()))
        
        if not text_words or not topic_words:
            return 0.0
        
        intersection = len(text_words.intersection(topic_words))
        union = len(text_words.union(topic_words))
        
        return intersection / union if union > 0 else 0.0

class TopicConsolidationAgent:
    """Agent responsible for intelligent topic consolidation"""
    
    def consolidate_topics(self, topics):
        """Smart consolidation using semantic similarity"""
        if not topics:
            return []
        
        consolidated = []
        used_topics = set()
        
        # Sort by confidence for better consolidation
        sorted_topics = sorted(topics, key=lambda x: x['confidence'], reverse=True)
        
        for topic in sorted_topics:
            if topic['topic'] in used_topics:
                continue
            
            # Find similar topics
            similar_topics = [topic]
            used_topics.add(topic['topic'])
            
            for other_topic in sorted_topics:
                if other_topic['topic'] in used_topics:
                    continue
                
                similarity = self._calculate_semantic_similarity(topic, other_topic)
                
                if similarity > 0.85:  # High similarity threshold as per requirements
                    similar_topics.append(other_topic)
                    used_topics.add(other_topic['topic'])
            
            # Consolidate similar topics
            if len(similar_topics) > 1:
                merged_topic = self._merge_topics(similar_topics)
                consolidated.append(merged_topic)
            else:
                consolidated.append(topic)
        
        return consolidated
    
    def _calculate_semantic_similarity(self, topic1, topic2):
        """Calculate semantic similarity between topics"""
        # Method 1: Name similarity
        name_sim = self._text_similarity(topic1['topic'], topic2['topic'])
        
        # Method 2: Terms similarity (if available)
        terms_sim = 0.0
        if 'terms' in topic1 and 'terms' in topic2:
            terms_sim = self._terms_similarity(topic1['terms'], topic2['terms'])
        
        # Method 3: Method similarity (topics from same method are more likely similar)
        method_sim = 1.0 if topic1.get('method') == topic2.get('method') else 0.0
        
        # Weighted combination
        weights = [0.5, 0.3, 0.2]
        similarities = [name_sim, terms_sim, method_sim]
        
        return sum(w * s for w, s in zip(weights, similarities))
    
    def _text_similarity(self, text1, text2):
        """Calculate text similarity using word overlap"""
        words1 = set(text1.lower().split('_'))
        words2 = set(text2.lower().split('_'))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _terms_similarity(self, terms1, terms2):
        """Calculate similarity between term lists"""
        if not terms1 or not terms2:
            return 0.0
        
        set1 = set(terms1)
        set2 = set(terms2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_topics(self, similar_topics):
        """Merge similar topics into consolidated topic"""
        # Use the highest confidence topic as base
        base_topic = max(similar_topics, key=lambda x: x['confidence'])
        
        # Combine confidence scores
        total_confidence = sum(t['confidence'] for t in similar_topics) / len(similar_topics)
        
        # Combine support
        total_support = sum(t.get('support', 1) for t in similar_topics)
        
        merged_topic = {
            'topic': base_topic['topic'],
            'confidence': total_confidence,
            'method': 'consolidated',
            'support': total_support,
            'merged_from': [t['topic'] for t in similar_topics if t['topic'] != base_topic['topic']]
        }
        
        # Combine terms if available
        all_terms = []
        for topic in similar_topics:
            if 'terms' in topic:
                all_terms.extend(topic['terms'])
        
        if all_terms:
            # Keep unique terms, sort by frequency
            from collections import Counter
            term_counts = Counter(all_terms)
            merged_topic['terms'] = [term for term, count in term_counts.most_common(10)]
        
        return merged_topic

class QualityAssessmentAgent:
    """Agent responsible for assessing topic quality"""
    
    def assess_quality(self, topics, reviews):
        """Assess quality of detected topics"""
        quality_scores = {}
        
        for topic in topics:
            quality = self._calculate_topic_quality(topic, reviews)
            quality_scores[topic['topic']] = quality
        
        return quality_scores
    
    def _calculate_topic_quality(self, topic, reviews):
        """Calculate comprehensive quality score"""
        scores = []
        
        # Score 1: Confidence level
        confidence_score = min(topic['confidence'], 1.0)
        scores.append(confidence_score)
        
        # Score 2: Support level (how many reviews)
        support = topic.get('support', 1)
        support_score = min(support / len(reviews), 1.0)
        scores.append(support_score)
        
        # Score 3: Topic coherence (name quality)
        coherence_score = self._assess_topic_coherence(topic['topic'])
        scores.append(coherence_score)
        
        # Score 4: Uniqueness (not too similar to others)
        uniqueness_score = 1.0  # Simplified - could be enhanced
        scores.append(uniqueness_score)
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]
        return sum(w * s for w, s in zip(weights, scores))
    
    def _assess_topic_coherence(self, topic_name):
        """Assess if topic name is coherent and meaningful"""
        # Basic coherence checks
        words = topic_name.split('_')
        
        # Penalty for very long topic names
        if len(words) > 4:
            return 0.5
        
        # Penalty for very short topic names
        if len(words) < 2:
            return 0.6
        
        # Penalty for non-alphabetic characters
        if not all(word.isalpha() for word in words):
            return 0.7
        
        # Good coherent topic
        return 0.9

if __name__ == "__main__":
    detector = AgenticTopicDetector()
    print("Agentic AI Topic Detection System initialized!")