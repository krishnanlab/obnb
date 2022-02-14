from tqdm import tqdm


class BaseFilter:
    """Base Filter object containing basic filter operations.

    Notes:
        Loop through all instances (IDs) retrieved by `self.get_ids` and decide
        whether or not to apply modification using `self.criterion`, and finally
        apply modification if passes criterion using `mod_fun`.

    Basic components (methods) needed for children filter classes:
        criterion: retrun true if the corresponding value of an instance passes
            the criterion
        get_ids: return list of IDs to scan through
        get_val_getter: return a function that map ID of an instance to some
            corresponding values
        get_mod_fun: return a function that modifies an instance

    All three 'get' methods above take a `LabelsetCollection` object as input

    """

    def __repr__(self):
        """Return name of the filer."""
        return self.__class__.__name__

    def __call__(self, lsc, progress_bar):
        entity_ids = self.get_ids(lsc)
        val_getter = self.get_val_getter(lsc)
        mod_fun = self.get_mod_fun(lsc)

        pbar = tqdm(entity_ids, disable=not progress_bar)
        pbar.set_description(f"{self!r}")
        for entity_id in pbar:
            if self.criterion(val_getter(entity_id)):
                mod_fun(entity_id)
