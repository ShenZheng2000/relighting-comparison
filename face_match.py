# import insightface
# import cv2
# import numpy as np
# import os

# # Load the pre-trained model
# model = insightface.app.FaceAnalysis()  # Use "buffalo_l" for the default model
# model.prepare(ctx_id=0, det_size=(640, 640))

# def split_concatenated_image(concatenated_image):
#     """
#     Split a concatenated image into three equal parts.
#     Assumes the image is horizontally concatenated as [image1 | image2 | image3].
#     """
#     height, width, _ = concatenated_image.shape
#     part_width = width // 3  # Divide the width into three equal parts
#     image1 = concatenated_image[:, :part_width, :]
#     image2 = concatenated_image[:, part_width:2 * part_width, :]
#     image3 = concatenated_image[:, 2 * part_width:, :]
#     return image1, image2, image3

# def get_face_embedding(image):
#     """
#     Detect and extract the face embedding from an image.
#     If no face is detected, return None.
#     """
#     faces = model.get(image)
#     if len(faces) == 0:
#         return None
#     return faces[0].normed_embedding  # Use the L2-normalized embedding

# def cosine_similarity(embedding1, embedding2):
#     """
#     Compute the cosine similarity between two embeddings.
#     """
#     return np.dot(embedding1, embedding2)

# def process_folder(folder_path):  # Remove the threshold parameter
#     """
#     Process all images in the folder, compute similarity between the second and third images,
#     and calculate the mean cosine similarity score.
#     """
#     similarity_scores = []  # List to store all similarity scores
#     no_face_count = 0  # Counter for images with no face detected

#     for filename in os.listdir(folder_path):
#         if filename.endswith(".png") or filename.endswith(".jpg"):  # Process only image files
#             concatenated_image_path = os.path.join(folder_path, filename)
#             concatenated_image = cv2.imread(concatenated_image_path)

#             if concatenated_image is None:
#                 print(f"Could not load image: {filename}")
#                 continue

#             # Split the concatenated image into three parts
#             image1, image2, image3 = split_concatenated_image(concatenated_image)

#             # Get face embeddings for the second and third images
#             embedding2 = get_face_embedding(image2)
#             embedding3 = get_face_embedding(image3)

#             if embedding2 is None or embedding3 is None:
#                 print(f"No face detected in one or both images: {filename}")
#                 no_face_count += 1
#                 continue

#             # Compute similarity between the second and third images
#             similarity = cosine_similarity(embedding2, embedding3)
#             similarity_scores.append(similarity)  # Add the similarity score to the list

#             print(f"Similarity between Image 2 and Image 3 in {filename}: {similarity}")

#     # Calculate the mean similarity score
#     if similarity_scores:  # Check if there are any valid similarity scores
#         mean_similarity = np.mean(similarity_scores)
#     else:
#         mean_similarity = None  # If no valid scores, set mean to None

#     print("\nSummary:")
#     print(f"Mean cosine similarity score: {mean_similarity}")
#     print(f"Number of images with no face detected: {no_face_count}")

# # Define the folder path and similarity threshold
# folder_path = "/home/shenzhen/Relight_Projects/relighting-comparison/outputs/exp_3_21_v3/candlelight"

# # Process the folder
# process_folder(folder_path)

# Mean cosine similarity score: 0.6321948170661926
# Number of images with no face detected: 47