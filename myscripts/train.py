import caffe
import numpy as np
import scipy.io
import math
import os
import sys
sys.path.append("..")
sys.path.append("../pylayers")
from pyutils import refine_util as rv

if __name__ == "__main__":
	caffe.set_mode_gpu()
	caffe.set_device(0)
	save_path = '../states/v3/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	solverproto = '../models/solver_v3.prototxt'
	solver = caffe.SGDSolver(solverproto)
	# default 0, need to be set when restart
	solver.set_iter(0) 
	# =============================
	Sov = rv.parse_solverproto(solverproto)
	max_iter = 60000;
	save_iter = 100;
	display_iter = 10
	_train_loss = 0
	tmpname = save_path + 'loss' + '.mat'
	cur_res_mat = save_path+'infer_res.mat'
	cur_iter = save_path+'iter.mat'
	

	if not os.path.exists(cur_iter):
		weights = '../models/pretrain/vgg16_20M.caffemodel'.format(save_path)
		solver.net.copy_from(weights)
		# solver.step(1)
		# solver.set_iter(1)
		solver.set_iter(0)
		begin = 1
		train_loss = np.zeros(int(math.ceil(max_iter/ display_iter)))
	else:
		curiter = scipy.io.loadmat(cur_iter)
		curiter = curiter['cur_iter']
		curiter = int(curiter)
		solver.set_iter(curiter)
		train_loss = scipy.io.loadmat(tmpname)
		train_loss = np.array(train_loss['train_loss'], dtype=np.float32).squeeze()
		solverstate = Sov['snapshot_prefix'] + \
		'_iter_{}.solverstate'.format(solver.iter)
		caffemodel = Sov['snapshot_prefix'] + \
		'_iter_{}.caffemodel'.format(solver.iter)

		if os.path.exists(solverstate):
			solver.restore(solverstate)
		elif os.path.exists(caffemodel):
			solver.net.copy_from(caffemodel)
		else:
			raise Exception("Model does not exist.")
		begin = solver.iter

	for iter in range(begin, max_iter):
		solver.step(1)
		_train_loss += solver.net.blobs['loss'].data
		if iter % display_iter == 0:
			train_loss[iter / display_iter] = _train_loss / display_iter
			_train_loss = 0
		if iter % save_iter == 0:
			batch, label, active = rv.getbatch(solver.net)
			scipy.io.savemat(cur_res_mat, dict(batch = batch, label = label, active = active))
			scipy.io.savemat(cur_iter, dict(cur_iter = iter))
			scipy.io.savemat(tmpname, dict(train_loss = train_loss))
			rv.clear_history(save_iter,Sov['snapshot_prefix'],iter)
