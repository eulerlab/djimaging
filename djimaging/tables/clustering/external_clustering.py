"""
This module defines the base class for external clustering tables.

Example usage:

from djimaging.tables import clustering

@schema
class RoiCluster(clustering.RoiClusterTemplate):
    roi_table = Roi
"""

from abc import abstractmethod

import datajoint as dj


class RoiClusterTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        clustering_id : tinyint unsigned
        -> self.roi_table
        ---
        cluster_idx : int
        """
        return definition

    @property
    @abstractmethod
    def roi_table(self):
        pass
