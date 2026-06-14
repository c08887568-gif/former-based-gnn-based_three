import os
import shutil
class _TraceDatasetmover:
    def __init__(self, filenames ,path, mode, data_type):
        self._filenames = filenames
        self._gnssfilenames = self._filenames['gnss']
        self._adjfilenames = self._filenames['adj']
        self._mode = mode
        self._data_type = data_type
        self._path_gnss = os.path.join(os.path.dirname(path['gnss']), path['gnss'].split('/')[-1]+'_'+ mode +'_'+ data_type)
        self._path_adj = os.path.join(os.path.dirname(path['adj']), path['adj'].split('/')[-1]+'_'+ mode +'_'+ data_type)
        os.makedirs(self._path_gnss, exist_ok=True)
        os.makedirs(self._path_adj, exist_ok=True)
    def move(self, index):
        # 源文件路径
        gnss_source = self._gnssfilenames[index]
        shutil.copy(gnss_source, self._path_gnss)
        if self._path_adj is not None:
            adj_source = self._adjfilenames[index]
            shutil.copy(adj_source, self._path_adj)
            return gnss_source + '移动完成' + '\n'+ adj_source +'移动完成'
        return gnss_source + '移动完成'