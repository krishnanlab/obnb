import pprint

import pandas as pd

from nleval.data.annotation.base import BaseAnnotationData
from nleval.typing import List, Optional


class GeneOntologyAnnotation(BaseAnnotationData):
    """Gene Ontology annotations.

    Gene ontology annotations are retrieved from geneontology.org.

    There are sevone categories of gene annotation evidences from the Gene
    Ontology. By default, we only use *Experimental evidences*, *Author
    Statements*, and the *Curator inferred* evidence types to ensure the
    quality of the annotations. See the gene ontology `evidence codes
    <http://geneontology.org/docs/guide-go-evidence-codes/>`_ documentation
    page for more information.

    - Experimental evidences (``EXPERIMENTAL``):
        - ``EXP`` Experiment
        - ``IDA`` Direct assay
        - ``IPI`` Physical interaction
        - ``IMP`` Mutant phenotype
        - ``IGI`` Genetic interaction
        - ``IEP`` Expression pattern

    - Phylogenetically-inffered (``PHYLOGENIC``):
        - ``IBA`` Biological aspect of ancestor
        - ``IBD`` Biological aspect of descendant
        - ``IKR`` Key residues
        - ``IRD`` Rapid divergence

    - Computational analysis (``COMPUTATIONAL``):
        - ``ISS`` Sequence or structural similarity
        - ``ISO`` Sequence orthology
        - ``ISA`` Sequence alignment
        - ``ISM`` Sequence model
        - ``IGC`` Genomic context
        - ``RCA`` Reviewed computational analysis

    - Author statements (``AUTHOR``):
        - ``TAS`` Tracable author statement
        - ``NAS`` Nontracable author statement

    - Curator statements (``CURATOR``):
        - ``IC`` Inferred by curator
        - ``ND`` No biological data available

    - Electronic annotation evidences (``ELECTRONIC``):
        - ``IEA`` Electronic annotation

    **[Last updated: 2023-03-10]**

    Args:
        root: Root directory of the data.
        data_sources: List of evidene types to be considered. If not set,
            then use the default channels (experimental evidences, author and
            curator statements).

    """

    annotation_file_name = "goa_human.gaf"
    annotation_url = "http://geneontology.org/gene-associations/goa_human.gaf.gz"

    def __init__(
        self,
        root: str,
        *,
        data_sources: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize GeneOntology annotation data object."""
        self._data_sources = data_sources
        super().__init__(root, **kwargs)

    @property
    def data_sources(self) -> List[str]:
        if self._data_sources is None:
            return [
                "EXP",  # Experiment
                "IDA",  # Direct Assay
                "IPI",  # Physical Interaction
                "IMP",  # Mutant Phenotype
                "IGI",  # Genetic Interaction
                "IEP",  # Expression Pattern
                "TAS",  # Traceable Author Statement
                "NAS",  # Non-traceable Author Statement
                "IC",  # Inferred by Curator
            ]
        else:
            return self._data_sources

    def process(self):
        in_path = self.raw_file_path(0)
        self.plogger.info(f"Loading raw annotation from {in_path}")

        # Load gene annotation data (gaf-version: 2.2)
        # http://geneontology.org/docs/go-annotation-file-gaf-format-2.2/
        annot_df = pd.read_csv(
            in_path,
            sep="\t",
            comment="!",
            header=0,
            names=[
                "db",
                "db_id",
                "db_symbol",
                "qual",
                "go_id",
                "db_ref",
                "ec",
                "wof",
                "aspect",
                "eb_name",
                "db_syn",
                "db_type",
                "taxon",
                "date",
                "assigned_by",
                "annot_ext",
                "gene_prod_id",
            ],
            low_memory=False,
        )

        # Select specified channels
        evidence_str = pprint.pformat(self.data_sources)
        self.plogger.info(f"Subsetting annotations to evidences:\n{evidence_str}")
        ind = annot_df["ec"].isin(self.data_sources)
        self.plogger.info(f"{ind.sum():,} (out of {ind.shape[0]:,}) entries selected")
        annot_df = annot_df[ind]

        # Bulk query gene symbol to entrez conversion and get converted genes
        gene_symbols_to_query = annot_df["db_symbol"].unique().tolist()
        gene_id_converter = self.get_gene_id_converter()
        gene_id_converter.query_bulk(gene_symbols_to_query)
        converted_symbols = {
            i for i in gene_symbols_to_query if gene_id_converter[i] is not None
        }
        ind = annot_df["db_symbol"].isin(converted_symbols)
        num_removed = ind.shape[0] - ind.sum()
        self.plogger.info(f"{num_removed:,} entries removed by gene id conversion.")
        annot_df = annot_df[ind]

        # Convert gene symbol to the desired gene id type
        # NOTE: assumes that the mappings are one-to-one
        annot_df["gene_id"] = annot_df["db_symbol"].apply(gene_id_converter.__getitem__)
        annot_df["term_id"] = annot_df["go_id"]

        # Save formatted annotation
        out_path = self.processed_file_path(0)
        self.plogger.info(f"Saving formatted annotation to {out_path}\n{annot_df}")
        annot_df[["gene_id", "term_id"]].to_csv(out_path, index=False)
