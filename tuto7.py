import numpy as np
from itertools import combinations

class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        """
        Initialize Apriori algorithm parameters.

        Parameters:
        - min_support: Minimum support threshold
        - min_confidence: Minimum confidence threshold
        """
        self.min_support = min_support                # Minimum required support
        self.min_confidence = min_confidence          # Minimum required confidence
        self.frequent_patterns = {}                   # Dictionary to store frequent itemsets
        self.rules = []                               # List to store association rules

    def _calculate_support(self, itemset, dataset):
        """Calculate the support value of an itemset within the dataset."""
        match_count = sum(1 for transaction in dataset if itemset.issubset(transaction))
        return match_count / len(dataset)

    def _generate_candidate_itemsets(self, current_itemsets, target_length):
        """Generate candidate itemsets of specified length from current frequent itemsets."""
        unique_items = sorted(set(item for itemset in current_itemsets for item in itemset))
        return list(combinations(unique_items, target_length))

    def _prune_candidates(self, candidate_list, previous_frequent_set, k):
        """Prune candidates that do not have all (k-1)-subsets in previous frequent itemsets."""
        pruned_list = []
        for candidate in candidate_list:
            subsets = combinations(candidate, k - 1)
            if all(frozenset(subset) in previous_frequent_set for subset in subsets):
                pruned_list.append(frozenset(candidate))
        return pruned_list

    def fit(self, transactions):
        """
        Main method to generate frequent itemsets and association rules.

        Parameters:
        - transactions: List of transactions (each transaction is a list of items)
        """
        # Convert transactions into frozensets for performance and consistency
        transaction_list = [frozenset(t) for t in transactions]

        # Extract all unique items from dataset
        unique_items = set(item for transaction in transaction_list for item in transaction)

        # Initialize candidate itemsets of size 1
        candidate_itemsets = [frozenset([item]) for item in unique_items]

        k = 1  # Start with itemsets of size 1

        # Iterate until no more frequent itemsets can be generated
        while candidate_itemsets:
            frequent_itemsets_k = []  # Frequent itemsets of current length

            # Check each candidate's support
            for itemset in candidate_itemsets:
                support = self._calculate_support(itemset, transaction_list)
                if support >= self.min_support:
                    frequent_itemsets_k.append((itemset, support))

            # If we found any frequent itemsets of this length
            if frequent_itemsets_k:
                self.frequent_patterns[k] = frequent_itemsets_k  # Store them
                k += 1

                # Generate next set of candidates from current frequent itemsets
                current_frequent_itemsets = [itemset for itemset, _ in frequent_itemsets_k]
                next_candidates = self._generate_candidate_itemsets(current_frequent_itemsets, k)

                # Convert to frozensets and prune them using Apriori principle
                prev_frequent_set = set(current_frequent_itemsets)
                candidate_itemsets = self._prune_candidates(next_candidates, prev_frequent_set, k)
            else:
                break  # Stop if no frequent itemsets found

        # Generate association rules from all frequent itemsets
        for length, itemsets in self.frequent_patterns.items():
            if length == 1:
                continue  # Rules cannot be formed from single items

            for itemset, itemset_support in itemsets:
                # Generate all non-empty subsets (antecedents)
                for i in range(1, length):
                    for left in combinations(itemset, i):
                        left_set = frozenset(left)
                        right_set = itemset - left_set

                        # Avoid rules with empty right-hand side
                        if not right_set:
                            continue

                        # Calculate confidence
                        left_support = self._calculate_support(left_set, transaction_list)
                        confidence = itemset_support / left_support

                        # Check confidence threshold
                        if confidence >= self.min_confidence:
                            self.rules.append({
                                'antecedent': left_set,
                                'consequent': right_set,
                                'support': itemset_support,
                                'confidence': confidence
                            })

    def get_frequent_itemsets(self, length=None):
        """
        Get discovered frequent itemsets.

        Parameters:
        - length: Specific size of itemsets to return (None returns all)

        Returns:
        - List of (itemset, support) tuples
        """
        if length is None:
            # Flatten all frequent itemsets
            return [itemset for sets in self.frequent_patterns.values() for itemset in sets]
        return self.frequent_patterns.get(length, [])

    def get_association_rules(self):
        """Return all valid association rules that meet min confidence."""
        return self.rules


# --------------------- Example Usage -------------------------
if __name__ == "__main__":
    # Sample market basket data
    sample_transactions = [
        ['bread', 'milk'],
        ['bread', 'diapers', 'beer', 'eggs'],
        ['milk', 'diapers', 'beer', 'cola'],
        ['bread', 'milk', 'diapers', 'beer'],
        ['bread', 'milk', 'diapers', 'cola']
    ]

    # Create Apriori model with specified thresholds
    model = Apriori(min_support=0.4, min_confidence=0.6)

    # Train model on transaction data
    model.fit(sample_transactions)

    # Display frequent itemsets
    print("Frequent Itemsets:")
    for k, itemsets in model.frequent_patterns.items():
        print(f"\nLength {k}:")
        for itemset, support in itemsets:
            print(f"{set(itemset)}: support = {support:.2f}")

    # Display association rules
    print("\nAssociation Rules:")
    for rule in model.get_association_rules():
        print(f"{set(rule['antecedent'])} => {set(rule['consequent'])} "
              f"(support={rule['support']:.2f}, confidence={rule['confidence']:.2f})")
