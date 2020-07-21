"""Learning embedding of graph using Poincare Ball Model."""

import logging

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.datasets.utils import load_karate_graph
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.expectation_maximization import RiemannianEM

DEFAULT_PLOT_PRECISION = 100

def plot_gaussian_mixture_distribution(data,
                                       mixture_coefficients,
                                       means,
                                       variances,
                                       plot_precision=DEFAULT_PLOT_PRECISION,
                                       save_path='',
                                       metric=None):
    """Plot Gaussian Mixture Model."""
    x_axis_samples = gs.linspace(-1, 1, plot_precision)
    y_axis_samples = gs.linspace(-1, 1, plot_precision)
    x_axis_samples, y_axis_samples = gs.meshgrid(x_axis_samples,
                                                 y_axis_samples)

    z_axis_samples = gs.zeros((plot_precision, plot_precision))

    for z_index, _ in enumerate(z_axis_samples):

        x_y_plane_mesh = gs.concatenate((
            gs.expand_dims(x_axis_samples[z_index], -1),
            gs.expand_dims(y_axis_samples[z_index], -1)),
            axis=-1)

        mesh_probabilities = PoincareBall.\
            weighted_gmm_pdf(
                mixture_coefficients,
                x_y_plane_mesh,
                means,
                variances,
                metric)

        z_axis_samples[z_index] = mesh_probabilities.sum(-1)

    fig = plt.figure('Learned Gaussian Mixture Model '
                     'via Expectation Maximization on Poincaré Disc')

    ax = fig.gca(projection='3d')
    ax.plot_surface(x_axis_samples,
                    y_axis_samples,
                    z_axis_samples,
                    rstride=1,
                    cstride=1,
                    linewidth=1,
                    antialiased=True,
                    cmap=plt.get_cmap("viridis"))
    z_circle = -0.8
    p = Circle((0, 0), 1,
               edgecolor='b',
               lw=1,
               facecolor='none')

    ax.add_patch(p)

    art3d.pathpatch_2d_to_3d(p,
                             z=z_circle,
                             zdir="z")

    for data_index, _ in enumerate(data):
        ax.scatter(data[data_index][0],
                   data[data_index][1],
                   z_circle,
                   c='b',
                   marker='.')

    for means_index, _ in enumerate(means):
        ax.scatter(means[means_index][0],
                   means[means_index][1],
                   z_circle,
                   c='r',
                   marker='D')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.8, 0.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')

    plt.savefig(save_path, format="pdf")

    return plt

def log_sigmoid(vector):
    """Logsigmoid function.

    Apply log sigmoid function

    Parameters
    ----------
    vector : array-like, shape=[n_samples, dim]

    Returns
    -------
    result : array-like, shape=[n_samples, dim]
    """
    return gs.log((1 / (1 + gs.exp(-vector))))


def grad_log_sigmoid(vector):
    """Gradient of log sigmoid function.

    Parameters
    ----------
    vector : array-like, shape=[n_samples, dim]

    Returns
    -------
    gradient : array-like, shape=[n_samples, dim]
    """
    return 1 / (1 + gs.exp(vector))


def grad_squared_distance(point_a, point_b):
    """Gradient of squared hyperbolic distance.

    Gradient of the squared distance based on the
    Ball representation according to point_a

    Parameters
    ----------
    point_a : array-like, shape=[n_samples, dim]
        First point in hyperbolic space.
    point_b : array-like, shape=[n_samples, dim]
        Second point in hyperbolic space.

    Returns
    -------
    dist : array-like, shape=[n_samples, 1]
        Geodesic squared distance between the two points.
    """
    hyperbolic_metric = PoincareBall(2).metric
    log_map = hyperbolic_metric.log(point_b, point_a)

    return -2 * log_map


def loss(example_embedding, context_embedding, negative_embedding,
         manifold):
    """Compute loss and grad.

    Compute loss and grad given embedding of the current example,
    embedding of the context and negative sampling embedding.
    """
    n_edges, dim =\
        negative_embedding.shape[0], example_embedding.shape[-1]
    example_embedding = gs.expand_dims(example_embedding, 0)
    context_embedding = gs.expand_dims(context_embedding, 0)

    positive_distance =\
        manifold.metric.squared_dist(
            example_embedding, context_embedding)
    positive_loss =\
        log_sigmoid(-positive_distance)

    reshaped_example_embedding =\
        gs.repeat(example_embedding, n_edges, axis=0)

    negative_distance =\
        manifold.metric.squared_dist(
            reshaped_example_embedding, negative_embedding)
    negative_loss = log_sigmoid(negative_distance)

    total_loss = -(positive_loss + negative_loss.sum())

    positive_log_sigmoid_grad =\
        -grad_log_sigmoid(-positive_distance)

    positive_distance_grad =\
        grad_squared_distance(example_embedding, context_embedding)

    positive_grad =\
        gs.repeat(positive_log_sigmoid_grad, dim, axis=-1)\
        * positive_distance_grad

    negative_distance_grad =\
        grad_squared_distance(reshaped_example_embedding, negative_embedding)

    negative_distance = gs.to_ndarray(negative_distance,
                                      to_ndim=2, axis=-1)
    negative_log_sigmoid_grad =\
        grad_log_sigmoid(negative_distance)

    negative_grad = negative_log_sigmoid_grad\
        * negative_distance_grad

    example_grad = -(positive_grad + negative_grad.sum(axis=0))

    return total_loss, example_grad


def main():
    """Learning Poincaré graph embedding.

    Learns Poincaré Ball embedding by using Riemannian
    gradient descent algorithm.
    """
    gs.random.seed(1234)
    dim = 2
    max_epochs = 20
    lr = .05
    n_negative = 2
    context_size = 1
    karate_graph = load_karate_graph()

    nb_vertices_by_edges =\
        [len(e_2) for _, e_2 in karate_graph.edges.items()]
    logging.info('Number of edges: %s', len(karate_graph.edges))
    logging.info(
        'Mean vertices by edges: %s',
        (sum(nb_vertices_by_edges, 0) / len(karate_graph.edges)))

    negative_table_parameter = 5
    negative_sampling_table = []

    for i, nb_v in enumerate(nb_vertices_by_edges):
        negative_sampling_table +=\
            ([i] * int((nb_v**(3. / 4.))) * negative_table_parameter)

    negative_sampling_table = gs.array(negative_sampling_table)
    random_walks = karate_graph.random_walk()
    embeddings = gs.random.normal(size=(karate_graph.n_nodes, dim))
    embeddings = embeddings * 0.2

    hyperbolic_manifold = PoincareBall(2)

    colors = {1: 'b', 2: 'r'}
    for epoch in range(max_epochs):
        total_loss = []
        for path in random_walks:

            for example_index, one_path in enumerate(path):
                context_index = path[max(0, example_index - context_size):
                                     min(example_index + context_size,
                                     len(path))]
                negative_index =\
                    gs.random.randint(negative_sampling_table.shape[0],
                                      size=(len(context_index),
                                      n_negative))
                negative_index = negative_sampling_table[negative_index]

                example_embedding = embeddings[one_path]

                for one_context_i, one_negative_i in zip(context_index,
                                                         negative_index):
                    context_embedding = embeddings[one_context_i]
                    negative_embedding = embeddings[one_negative_i]
                    l, g_ex = loss(
                        example_embedding,
                        context_embedding,
                        negative_embedding,
                        hyperbolic_manifold)
                    total_loss.append(l)

                    example_to_update = embeddings[one_path]
                    embeddings[one_path] = hyperbolic_manifold.metric.exp(
                        -lr * g_ex, example_to_update)

        logging.info(
            'iteration %d loss_value %f',
            epoch, sum(total_loss, 0) / len(total_loss))

    import pickle
    with open('kirikou2', 'wb') as f:
        pickle.dump(embeddings, f)


    circle = visualization.PoincareDisk(point_type='ball')
    plt.figure()
    ax = plt.subplot(111)
    circle.add_points(gs.array([[0, 0]]))
    circle.set_ax(ax)
    circle.draw(ax=ax)
    for i_embedding, embedding in enumerate(embeddings):
        plt.scatter(
            embedding[0], embedding[1],
            c=colors[karate_graph.labels[i_embedding][0]])
    plt.show()

    em = RiemannianEM(
        n_gaussians=2,
        riemannian_metric=hyperbolic_manifold.metric,
        initialisation_method='random',
        mean_method='frechet-poincare-ball',
    )

    means, variances, mixture_coefficients, posterior_probs = em.fit(
        data=embeddings,
        max_iter=100)

    plot = plot_gaussian_mixture_distribution(embeddings,
                                              mixture_coefficients,
                                              means,
                                              variances,
                                              plot_precision=100,
                                              save_path='result.png',
                                              metric=hyperbolic_manifold.metric)

    plot.show()

if __name__ == '__main__':
    main()
