from backend.router.query_router import QueryRouter, QueryCategory


def test_query_router_classifications():
    router = QueryRouter()

    theory_query = "What are the four necessary conditions for deadlock?"
    res_theory = router.route(theory_query)
    assert res_theory.category == QueryCategory.THEORY

    numerical_query = "Given the allocation matrix and max matrix, calculate the safe sequence using Banker's algorithm."
    res_numerical = router.route(numerical_query)
    assert res_numerical.category == QueryCategory.NUMERICAL

    graph_query = "Find if there is a cycle in this wait-for graph."
    res_graph = router.route(graph_query)
    assert res_graph.category == QueryCategory.GRAPH

    lab_query = "How to write a pthread program demonstrating deadlock with mutex locks?"
    res_lab = router.route(lab_query)
    assert res_lab.category == QueryCategory.LAB
