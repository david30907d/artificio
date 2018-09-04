"""

 similar_images_TL.py  (author: Anson Wong / git: ankonzoid)

 We find similar images in a database by using transfer learning
 via a pre-trained VGG image classifier. We plot the 5 most similar
 images for each image in the database, and plot the tSNE for all
 our image feature vectors.

"""
import sys, os
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from keras.models import Model

from src.plot_utils import plot_query_answer
from src.sort_utils import find_topk_unique
from src.kNN import kNN
from src.tSNE import plot_tsne

from pathlib import Path
import tqdm


def main():
    # ================================================
    # Load pre-trained model and remove higher level layers
    # ================================================
    print("Loading inception_resnet_v2 pre-trained model...")
    model = InceptionResNetV2(weights='imagenet', include_top=False) 

    # ================================================
    # Read images and convert them to feature vectors
    # ================================================
    imgs, filename_heads = [], []
    img_list = []
    path = "db"
    print("Reading images from '{}' directory...\n".format(path))
    for index, f in enumerate(Path(path).iterdir()):
        if index == 50:
            break
        # Process filename
        head, ext = f.name, f.suffix
        if ext.lower() not in [".jpg", ".jpeg"]:
            continue

        # Read image file
        img = image.load_img(f, target_size=(224, 224))  # load
        imgs.append(np.array(img))  # image
        filename_heads.append(head)  # filename head

        # Pre-process for model input
        img = image.img_to_array(img)  # convert to array
        img = np.expand_dims(img, axis=0)
        if len(img_list) > 0:
            img_list = np.concatenate((img_list, img))
        else:
            img_list = img

    img_list = preprocess_input(img_list)
    X = model.predict(img_list).reshape(len(img_list), -1)
    imgs = np.array(imgs)  # images
    print("imgs.shape = {}".format(imgs.shape))
    print("X_features.shape = {}\n".format(X.shape))

    # ===========================
    # Find k-nearest images to each image
    # ===========================
    n_neighbours = 5 + 1  # +1 as itself is most similar
    knn = kNN()  # kNN model
    knn.compile(n_neighbors=n_neighbours, algorithm="brute", metric="cosine")
    knn.fit(X)

    # ==================================================
    # Plot recommendations for each image in database
    # ==================================================
    output_rec_dir = Path('output', 'rec')
    if not output_rec_dir.exists():
        output_rec_dir.mkdir()
    n_imgs = len(imgs)
    ypixels, xpixels = imgs[0].shape[0], imgs[0].shape[1]
    for ind_query in range(n_imgs):

        # Find top-k closest image feature vectors to each vector
        print("[{}/{}] Plotting similar image recommendations for: {}".format(ind_query+1, n_imgs, filename_heads[ind_query]))
        distances, indices = knn.predict(np.array([X[ind_query]]))
        distances = distances.flatten()
        indices = indices.flatten()
        indices, distances = find_topk_unique(indices, distances, n_neighbours)

        # Plot recommendations
        rec_filename = Path(output_rec_dir, "{}_rec.png".format(filename_heads[ind_query]))
        x_query_plot = imgs[ind_query].reshape((-1, ypixels, xpixels, 3))
        x_answer_plot = imgs[indices].reshape((-1, ypixels, xpixels, 3))
        plot_query_answer(x_query=x_query_plot,
                          x_answer=x_answer_plot[1:],  # remove itself
                          filename=rec_filename)

    # ===========================
    # Plot tSNE
    # ===========================
    output_tsne_dir = Path("output")
    if not output_tsne_dir.exists():
        output_tsne_dir.mkdir()
    tsne_filename = Path(output_tsne_dir, "tsne.png")
    print("Plotting tSNE to {}...".format(tsne_filename))
    plot_tsne(imgs, X, tsne_filename)

# Driver
if __name__ == "__main__":
    main()