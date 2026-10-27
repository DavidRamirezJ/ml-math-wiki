"""Inner product demo — real NumPy output shown in the video. Toy numbers only (except the book's attention example)."""
import numpy as np

x, y = np.array([1, 2, 3]), np.array([4, -5, 6])
xy = x @ y                                         # dot product: sum of x_i * y_i
a, b = np.array([1, 1]), np.array([1, 2])
ab = a @ b
cos_ab = ab / (np.sqrt(a @ a) * np.sqrt(b @ b))    # "length" = sqrt(x . x) (video 6)
deg_ab = np.degrees(np.arccos(cos_ab))
orth = np.array([1, 1]) @ np.array([-1, 1])
words = {"cat": np.array([0.9, 0.8, 0.1]),         # toy word vectors (NOT real embeddings)
         "dog": np.array([0.8, 0.9, 0.2]),
         "car": np.array([0.1, 0.2, 0.9])}


def cos_sim(p, q):
    return p @ q / (np.sqrt(p @ p) * np.sqrt(q @ q))


cat_dog = cos_sim(words["cat"], words["dog"])
cat_car = cos_sim(words["cat"], words["car"])
your = np.array([0.43, 0.15, 0.89])                # book example (Raschka §3.3.1, p. 58)
journey = np.array([0.55, 0.87, 0.66])
score = your @ journey

if __name__ == "__main__":
    print("x . y =", xy)
    print("(1,1) . (1,2) =", ab)
    print("cos(angle) =", round(cos_ab, 4), "  angle =", round(deg_ab, 1), "degrees")
    print("(1,1) . (-1,1) =", orth, "  -> orthogonal")
    print("symmetric?", x @ y == y @ x, "  x . x =", x @ x, "> 0")
    print("cos_sim(cat, dog) =", round(cat_dog, 2))
    print("cos_sim(cat, car) =", round(cat_car, 2))
    print("score(Your, journey) =", round(score, 4))
