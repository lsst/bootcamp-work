
from astropy.table import Table, Column

def build_cluster_table(cluster_results_dict, data_types):
    """
    """
    col_list = []
    for key, dtype in data_types.items():
        results_data = cluster_results_dict.get(key)
        if results_data is None:
            continue
        col = Column(results_data, name=key, dtype=dtype)
        col_list.append(col)
    ret_table = Table(col_list)
    return ret_table



