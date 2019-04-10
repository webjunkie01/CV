# import necessary packages

import hdidx
import numpy as np




# print log messages
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# generating sample data
ndim = 256     # dimension of features
ndb = 10000    # number of dababase items
nqry = 120     # number of queries


X_db = np.random.random((ndb, ndim)).astype(np.float64)
X_qry = np.random.random((nqry, ndim)).astype(np.float32)

# create Product Quantization Indexer
idx = hdidx.indexer.PQIndexer()
# set storage for the indexer, this time we store the indexes into LMDB
idx.set_storage('lmdb', {'path': '/tmp/pq.idx'})
# build indexer
idx.build({'vals': X_db, 'nsubq': 8})
# save the index information for future use
idx.save('/tmp/pq.info')
# add database items to the indexer
idx.add(X_db)
# searching in the database, and return top-100 items for each query
idx.search(X_qry, 100)

