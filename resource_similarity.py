import argparse
import base64
import logging
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from requests.exceptions import HTTPError, RequestException, Timeout
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

# Logging setup; this will write to STDOUT and a DEBUG-level log to resourceSimilarity.log
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler("resource_similarity.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

include_properties = [
    "name",
    "entityTypes",
    "_references",
]
ignore_properties = [
    "_compositeId",
    "_compositeOfIds",
    "_createdAt",
    "_executionTime",
    "_id",
    "_modifiedAt",
    "_observedAt",
    "_startedAt",
    "_timeSinceLastUpdate",
    "accessScopeTokens",
    "beginTime",
    "changeTime",
    "createTime",
    "entityChangeTime",
    "entityCreateTime",
    "geolocation",
    "groupPerTokenRule",
    "groupTokens",
    "historyExcludeTokens",
    "lastBroadcastTime",
    "matchTokens",
    "mergeTokens",
    "observedTime",
    "tags",
    "uniqueId",
    "vertexType",
]


class ResourceNotFoundError(Exception):
    """Custom class for handling 404s, just passes.

    Args:
        Exception: Thrown if a resource can't be found.
    """

    pass


def get_header(username, password):
    """Builds headers for Topology Manager including the
    default tenant ID and Base64 encoded basic auth.

    Args:
        username (str): Username to use.
        password (str): Password to use.

    Returns:
        dict: The headers to use.
    """
    auth = username + ":" + password
    basicAuth = base64.b64encode(auth.encode())
    header = {
        "Authorization": "Basic " + basicAuth.decode(),
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-TenantID": "cfd95b7e-3bc7-4006-a4a8-a73a79c71255",
        "User-Agent": "resource_similarity.py",
    }
    logging.info(f"Headers are: {header}")
    return header


@retry(
    stop=stop_after_attempt(4),  # maximum number of retries
    wait=wait_exponential(multiplier=5, min=4, max=5),  # exponential backoff
)
def get_resource_data(server, headers, _id):
    """Gets the specified resource from the Topology Manager APIs
    including all properties and its relationships.

    Args:
        server (str): Server to query.
        headers (dict): The header payload to use.
        _id (str): The resource _id to query for.
        retries (int, optional): Number of retries. Defaults to 3.
        timeout (int, optional): Timeout. Defaults to 5.

    Raises:
        ResourceNotFoundError: If a 404 is received.

    Returns:
        json: Resource data for the specified _id.
    """
    url = f"https://{server}/1.0/topology/resources/{_id}?_field=*&_relation=*"
    logger.info(f"Fetching data from {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        raise ResourceNotFoundError(f"Resource not found at {url} (HTTP 404 response)")
    response.raise_for_status()
    return response.json()


def get_resource_features(resource, include, exclude):
    """Extracts features from the specified resource by including or ignoring
    specified properties. Any relationships that are not ignored are summarised
    by their type and sorted.

    Args:
        resource (dict): The resource to extract features from.
        include_properties (list): If specified, properties to include in features.
        ignore_properties (list): If specified, properties to exclude from features.

    Returns:
        dict: Features of the specified resource.
    """
    _id = resource["_id"]
    features = {}
    if include and not exclude:
        features = {k: v for k, v in resource.items() if k in include}
    elif exclude and not include:
        features = {k: v for k, v in resource.items() if k not in exclude}
    if "_references" in features:
        edges = features["_references"]
        del features["_references"]
        edgeTypes = []
        for edge in edges:
            edgeTypes.append(edge["_edgeType"])
        features["_references"] = sorted(set(edgeTypes))
    logger.info(f"{_id} features: {features}")
    return features


def get_cosine_similarity(data):
    """Calculates the cosine similarity of the specified data
    and returns the matrix and score to 2 decimal places.

    Args:
        data (list of dict): The data to process.

    Returns:
        ndarray: The calculated matrix.
        score: The similarity scores.
    """
    vec = DictVectorizer(sparse=False)
    vectors = vec.fit_transform(data)
    matrix = cosine_similarity(vectors)
    if len(matrix) == 2:
        return matrix, round(matrix[0, 1], 2)
    else:
        return matrix, np.round(matrix, 2)


def plot_heatmap(data, matrix, title="Cosine Similarity of Resources"):
    """Plots a heatmap of the specified matrix using the specified data for labels
    and the specified title.

    Args:
        data (list of dict): The data to derive labels X and Y axis labels from.
        matrix (ndarray): The data for the heatmap.
        title (str, optional): The heatmap title. Defaults to "Cosine Similarity of Resources".
    """
    labels = [f"{data[i]['name']}" for i in range(len(data))]
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix,
        cmap="Blues",
        annot=True,
        fmt="0.3f",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title(title)
    plt.show()


def main():
    """
    Run the application...
    """
    logging.info("Arguments: " + str(sys.argv))
    try:
        parser = argparse.ArgumentParser(
            description="Determines the similarity of the specified Topology Manager resources."
        )
        parser.add_argument(
            "-c",
            "--compare",
            dest="compare",
            type=str,
            help="Compare the specified set of resource _id values. Use multiple times for an n-way compare, minimum is 2 values.",
            required=True,
            action="append",
        )
        parser.add_argument(
            "-s",
            "--server",
            dest="server",
            type=str,
            help="The Topology Manager topology service server to query, e.g. your-topology-server",
            required=True,
        )
        parser.add_argument(
            "-u",
            "--user",
            dest="user",
            type=str,
            help="The username to query the Topology Manager server as, e.g. aiops-topology-cp4waiops-user",
            required=True,
        )
        parser.add_argument(
            "-p",
            "--password",
            dest="password",
            type=str,
            help="The password associated with the specified Topology Manager username, e.g. mypassword",
            required=True,
        )
        parser.add_argument(
            "-v",
            "--visualize",
            dest="visualize",
            action="store_true",
            help="Show a heatmap showing resource similarity, default is false.",
        )
        parser.add_argument(
            "-i",
            "--include",
            dest="include",
            action="store_true",
            help="Only use the name, entityTypes, tags, matchTokens, mergeTokens and summarised _references properties.",
        )
        parser.set_defaults(heatmap=False)

        args = parser.parse_args()
        logger.info(f"main - arguments: {args}")
        if len(args.compare) >= 2:
            header = get_header(args.user, args.password)
            ids = sorted(set(args.compare))
            logger.info("Getting resources:- %s", ids)
            resources = []
            for _id in ids:
                resource = get_resource_data(args.server, header, _id)
                if resource:
                    features = None
                    if args.include:
                        features = get_resource_features(
                            resource, include_properties, None
                        )
                    else:
                        features = get_resource_features(
                            resource, None, ignore_properties
                        )
                    resources.append(features)
                else:
                    logger.error(f"Could not find a resource with _id:- {_id}")
                    sys.exit(1)
            matrix, similarity_score = get_cosine_similarity(resources)
            logger.info(
                f"Similarity score of the specified resources:-\n{similarity_score}"
            )
            if args.visualize and len(matrix) > 0:
                plot_heatmap(resources, matrix)
        else:
            logger.error(
                "There must be at-least two resource _id values specified to compare similarity."
            )
            sys.exit(1)
    except Exception as e:
        logger.error(f"An error occured: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
