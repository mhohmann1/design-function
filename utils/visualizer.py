import matplotlib.pyplot as plt
from sklearn import manifold
from args import args

def save_single_image(points_pred, points_true=None, filename="point_cloud.png", min=None, max=None):
    if min is not None and max is not None:
        points_pred = points_pred * (max - min) + min
        if points_true is not None:
            points_true = points_true * (max - min) + min
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_pred[:, 0], points_pred[:, 1], points_pred[:, 2], c="grey", label="Reconstructed")
    if points_true is not None:
        ax.scatter(points_true[:, 0], points_true[:, 1], points_true[:, 2], c="blue", alpha=0.7, label="Ground Truth")
    ax.set_xlabel("X in mm", fontsize=14, labelpad=10)
    ax.set_ylabel("Y in mm", fontsize=14, labelpad=10)
    ax.set_zlabel("Z in mm", fontsize=14, labelpad=10)
    ax.set_zlim(0, 50)
    # ax.set_title("3D Point Cloud", fontsize=16, pad=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if points_true is not None:
        ax.legend(loc="upper right",  fontsize=12)
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def save_as_row(points_pred, points_true=None, filename="point_cloud.png", min=None, max=None):
    if min is not None and max is not None:
        points_pred = points_pred * (max - min) + min
        if points_true is not None:
            points_true = points_true * (max - min) + min

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))

    # 3D View
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.scatter(points_pred[:, 0], points_pred[:, 1], points_pred[:, 2], c="grey", label="Reconstructed")
    if points_true is not None:
        ax1.scatter(points_true[:, 0], points_true[:, 1], points_true[:, 2], c="turquoise", alpha=0.5, label="True")
    ax1.set_xlabel("X in mm", fontsize=16, labelpad=12)
    ax1.set_ylabel("Y in mm", fontsize=16, labelpad=12)
    ax1.set_zlabel("Z in mm", fontsize=16, labelpad=12)
    ax1.set_title("3D View", fontsize=18, pad=15)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(loc="upper right", fontsize=14)

    # XY Projection
    ax2 = axes[1]
    ax2.scatter(points_pred[:, 0], points_pred[:, 1], c="grey", label="Reconstructed")
    if points_true is not None:
        ax2.scatter(points_true[:, 0], points_true[:, 1], c="turquoise", alpha=0.5, label="True")
    ax2.set_xlabel("X in mm", fontsize=16, labelpad=12)
    ax2.set_ylabel("Y in mm", fontsize=16, labelpad=12)
    ax2.set_title("XY Projection", fontsize=18, pad=12)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(loc="upper right", fontsize=14)

    # YZ Projection
    ax3 = axes[2]
    ax3.scatter(points_pred[:, 1], points_pred[:, 2], c="grey", label="Reconstructed")
    if points_true is not None:
        ax3.scatter(points_true[:, 1], points_true[:, 2], c="turquoise", alpha=0.5, label="True")
    ax3.set_xlabel("Y in mm", fontsize=16, labelpad=12)
    ax3.set_ylabel("Z in mm", fontsize=16, labelpad=12)
    ax3.set_title("YZ Projection", fontsize=18, pad=12)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.legend(loc="upper right", fontsize=14)

    # XZ Projection
    ax4 = axes[3]
    ax4.scatter(points_pred[:, 0], points_pred[:, 2], c="grey", label="Reconstructed")
    if points_true is not None:
        ax4.scatter(points_true[:, 0], points_true[:, 2], c="turquoise", alpha=0.5, label="True")
    ax4.set_xlabel("X in mm", fontsize=16, labelpad=12)
    ax4.set_ylabel("Z in mm", fontsize=16, labelpad=12)
    ax4.set_title("XZ Projection", fontsize=18, pad=12)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.legend(loc="upper right", fontsize=14)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def save_as_image(points_pred, points_true=None, filename="point_cloud.png", min=None, max=None):

    if min is not None and max is not None:
        points_pred = points_pred * (max - min) + min
        if points_true is not None:
            points_true = points_true * (max - min) + min

    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(points_pred[:, 0], points_pred[:, 1], points_pred[:, 2], c="grey", label="Predicted")
    if points_true is not None:
        ax1.scatter(points_true[:, 0], points_true[:, 1], points_true[:, 2], c="yellow", alpha=0.3, label="True")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D View")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(222)
    ax2.scatter(points_pred[:, 0], points_pred[:, 1], c="grey", label="Predicted")
    if points_true is not None:
        ax2.scatter(points_true[:, 0], points_true[:, 1], c="yellow", alpha=0.3, label="True")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("XY Projection")
    ax2.legend(loc="upper right")


    ax3 = fig.add_subplot(223)
    ax3.scatter(points_pred[:, 1], points_pred[:, 2], c="grey", label="Predicted")
    if points_true is not None:
        ax3.scatter(points_true[:, 1], points_true[:, 2], c="yellow", alpha=0.3, label="True")

    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    ax3.set_title("YZ Projection")
    ax3.legend(loc="upper right")

    ax4 = fig.add_subplot(224)
    ax4.scatter(points_pred[:, 0], points_pred[:, 2], c="grey", label="Predicted")
    if points_true is not None:
        ax4.scatter(points_true[:, 0], points_true[:, 2], c="yellow", alpha=0.3, label="True")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Z")
    ax4.set_title("XZ Projection")
    ax4.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def visualize_z(latent_z, filename="latent_space.png"):
    tsne = manifold.TSNE(n_components=2, random_state=args.seed)
    z_transformed = tsne.fit_transform(latent_z)

    plt.figure(figsize=(8, 8))
    plt.scatter(z_transformed[:, 0], z_transformed[:, 1], s=5, alpha=0.7)
    plt.title("Latent Space Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    plt.savefig(filename, format="png", dpi=300)
    plt.close()

def plot_loss(train_loss, test_loss, label, save_img=False, show_img=False, path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label=f"Training {label}")
    plt.plot(test_loss, label=f"Testing {label}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{label}")
    plt.legend(loc="upper right")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    if save_img:
        plt.savefig(path)
    if show_img:
        plt.show()
    plt.close()