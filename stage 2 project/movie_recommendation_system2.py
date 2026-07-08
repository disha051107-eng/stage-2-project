import numpy as np
import pandas as pd


class FromScratchCollaborativeFilter:

    def __init__(self, k_neighbors=2):
        self.k_neighbors = k_neighbors
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.mean_user_ratings = None

    def fit(self, df, user_col, item_col, rating_col):
        """Builds the interaction matrix and calculates cosine similarity from scratch."""
        # 1. Build User-Item Interaction Matrix
        self.user_item_matrix = df.pivot(
            index=user_col, columns=item_col, values=rating_col
        )

        # Store mean ratings for centering (handles user rating bias)
        self.mean_user_ratings = self.user_item_matrix.mean(axis=1)

        # Normalize ratings by subtracting the mean (Mean-Centering)
        # We fill NaNs with 0 *after* centering so unrated items don't skew the mean
        centered_matrix = self.user_item_matrix.sub(
            self.mean_user_ratings, axis=0
        ).fillna(0)

        # Convert to numpy array for raw linear algebra operations
        R = centered_matrix.to_numpy()

        # 2. Compute Cosine Similarity Matrix Mathematically
        # Formula: (A dot B) / (norm(A) * norm(B))
        dot_product = np.dot(R, R.T)
        norms = np.linalg.norm(R, axis=1, keepdims=True)

        # Avoid division by zero for users with zero variance/no ratings
        norms[norms == 0] = 1e-9

        # Matrix multiplication of norms to get all pair-wise norm products
        norm_product = np.dot(norms, norms.T)

        # Compute final similarity matrix
        similarity_array = dot_product / norm_product

        # Convert back to DataFrame for easy indexing
        self.user_similarity_matrix = pd.DataFrame(
            similarity_array,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index,
        )

    def predict_rating(self, user, item):
        """Predicts the rating for a target user and item using K-Nearest Neighbors."""
        # Check if item exists in training data
        if item not in self.user_item_matrix.columns:
            return self.mean_user_ratings.get(user, 0)

        # Get top similar users who have actually rated the target item
        users_who_rated = self.user_item_matrix[
            self.user_item_matrix[item].notna()
        ].index

        if user in users_who_rated:
            # If they already rated it, return the actual rating
            return self.user_item_matrix.loc[user, item]

        # Get similarities of other users with the target user
        similarities = self.user_similarity_matrix.loc[user, users_who_rated]

        # Sort and select Top-K nearest neighbors
        top_k_neighbors = similarities.nlargest(self.k_neighbors)

        if top_k_neighbors.sum() == 0:
            # Fallback to the user's average rating if no valid neighbors exist
            return self.mean_user_ratings.loc[user]

        # Fetch neighbor ratings and their corresponding means
        neighbor_ratings = self.user_item_matrix.loc[top_k_neighbors.index, item]
        neighbor_means = self.mean_user_ratings.loc[top_k_neighbors.index]

        # Normalized rating = neighbor_rating - neighbor_mean
        normalized_neighbor_ratings = neighbor_ratings - neighbor_means

        # Weighted sum prediction formula
        weighted_sum = np.dot(top_k_neighbors, normalized_neighbor_ratings)
        sum_of_similarities = np.abs(top_k_neighbors).sum()

        # Target user mean + weighted baseline deviation
        predicted_rating = (
            self.mean_user_ratings.loc[user] + (weighted_sum / sum_of_similarities)
        )

        return np.clip(predicted_rating, 1.0, 5.0)  # Clip to standard 1-5 scale

    def get_recommendations(self, user, top_n=3):
        """Finds missing ratings, predicts them, and returns top N recommendations."""
        if user not in self.user_item_matrix.index:
            return "User not found."

        # Find items the user has NOT rated yet
        user_ratings = self.user_item_matrix.loc[user]
        unviewed_items = user_ratings[user_ratings.isna()].index

        predictions = []
        for item in unviewed_items:
            pred_rating = self.predict_rating(user, item)
            predictions.append((item, round(pred_rating, 2)))

        # Sort recommendations by highest predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:top_n]


# ==========================================
# TEST IMPLEMENTATION (Using Dummy Data)
# ==========================================
if __name__ == "__main__":
    # Mock Dataset: Movie Ratings (User, Movie, Rating)
    raw_data = {
        "User": [
            "Alice",
            "Alice",
            "Alice",
            "Bob",
            "Bob",
            "Bob",
            "Charlie",
            "Charlie",
            "Charlie",
            "David",
            "David",
        ],
        "Movie": [
            "Inception",
            "The Matrix",
            "Interstellar",
            "Inception",
            "The Matrix",
            "Toy Story",
            "The Matrix",
            "Interstellar",
            "Toy Story",
            "Inception",
            "Toy Story",
        ],
        "Rating": [5, 4, 1, 5, 5, 2, 2, 4, 5, 4, 1],
    }

    df = pd.DataFrame(raw_data)
    print("--- Raw Input Data ---")
    print(df)

    # Initialize and Train Engine
    engine = FromScratchCollaborativeFilter(k_neighbors=4)
    engine.fit(df, user_col="User", item_col="Movie", rating_col="Rating")

    print("\n--- User-Item Interaction Matrix ---")
    print(engine.user_item_matrix)

    print("\n--- Mathematically Computed Cosine Similarity Matrix ---")
    print(engine.user_similarity_matrix.round(4))

    # Generate Recommendations
    target_user = input("Enter user name: ")
recs = engine.get_recommendations(user=target_user, top_n=7)

print(f"\n--- Top Recommendations for {target_user} ---")
for movie, score in recs:
    print(f"Movie: {movie} | Predicted Rating: {score}")