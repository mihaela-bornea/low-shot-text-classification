import pickle
import os.path


class EmbeddingsCache():

    def __init__(self, cache_filename):
        self.cache_filename = cache_filename
        self._read_cache()

    def close(self, update=True):
        if update:
            self._dump_cache()
        self.cache = None

    def __contains__(self, text):
        return text in self.cache

    def add_entry(self, text, rep):
        self.cache[text] = rep
        if len(self.cache) % 10000 == 0:
            print('cache size', len(self.cache))

    def text2rep(self, text):
        return self.cache[text]

    def _read_cache(self):
        if os.path.isfile(self.cache_filename):
            with open(self.cache_filename, 'rb') as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}

    def _dump_cache(self):
        with open(self.cache_filename, 'wb') as f:
            pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)

