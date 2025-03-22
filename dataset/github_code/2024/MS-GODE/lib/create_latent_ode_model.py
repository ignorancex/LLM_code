from lib.gnn_models import GNN, maskGNN,Node_GCN,GNN_cgode
from lib.latent_ode import LatentGraphODE, LatentODE, CoupledODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver,GraphODEFunc,DiffeqSolver_lode,CoupledODEFunc,DiffeqSolver_cgode
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
def create_LatentODE_mask_model(args, input_dim, z0_prior, obsrv_std, device):
	# dim related
	latent_dim = args.latents # ode output dimension
	rec_dim = args.rec_dims
	input_dim = input_dim
	ode_dim = args.ode_dims #ode gcn dimension

	#encoder related
	encoder_z0 = maskGNN(in_dim=input_dim, n_hid=rec_dim, out_dim=latent_dim, n_heads=args.n_heads,
						 n_layers=args.rec_layers, dropout=args.dropout_mask, conv_name=args.z0_encoder,
						 aggregate=args.rec_attention,thresholding=args.thresholding)  # [b,n_ball,e]

	#ODE related
	if args.augment_dim > 0:
		ode_input_dim = latent_dim + args.augment_dim
	else:
		ode_input_dim = latent_dim

	ode_func_net = maskGNN(in_dim = ode_input_dim,n_hid =ode_dim,out_dim = ode_input_dim,n_heads=args.n_heads,n_layers=args.gen_layers,dropout=args.dropout_mask,conv_name = args.odenet,aggregate="add",thresholding=args.thresholding)

	gen_ode_func = GraphODEFunc(
		ode_func_net=ode_func_net,
		device=device).to(device)

	diffeq_solver = DiffeqSolver(gen_ode_func, args.solver, args=args,odeint_rtol=1e-2, odeint_atol=1e-2, device=device)

    #Decoder related
	decoder = maskDecoder(latent_dim, input_dim,thresholding=args.thresholding).to(device)


	model = LatentGraphODE(
		input_dim = input_dim,
		latent_dim = args.latents,
		encoder_z0 = encoder_z0,
		decoder = decoder,
		diffeq_solver = diffeq_solver,
		z0_prior = z0_prior,
		device = device,
		obsrv_std = obsrv_std,
		).to(device)

	return model



def create_LatentGODE_model(args, input_dim, z0_prior, obsrv_std, device):
	# dim related
	latent_dim = args.latents # ode output dimension
	rec_dim = args.rec_dims
	input_dim = input_dim
	ode_dim = args.ode_dims #ode gcn dimension

	#encoder related
	encoder_z0 = GNN(in_dim=input_dim, n_hid=rec_dim, out_dim=latent_dim, n_heads=args.n_heads,
						 n_layers=args.rec_layers, dropout=args.dropout, conv_name=args.z0_encoder,
						 aggregate=args.rec_attention)  # [b,n_ball,e]

	#ODE related
	if args.augment_dim > 0:
		ode_input_dim = latent_dim + args.augment_dim
	else:
		ode_input_dim = latent_dim

	ode_func_net = GNN(in_dim = ode_input_dim,n_hid =ode_dim,out_dim = ode_input_dim,n_heads=args.n_heads,n_layers=args.gen_layers,dropout=args.dropout,conv_name = args.odenet,aggregate="add")

	gen_ode_func = GraphODEFunc(
		ode_func_net=ode_func_net,
		device=device).to(device)

	diffeq_solver = DiffeqSolver(gen_ode_func, args.solver, args=args,odeint_rtol=1e-2, odeint_atol=1e-2, device=device)

    #Decoder related
	decoder = Decoder(latent_dim, input_dim).to(device)

	model = LatentGraphODE(
		input_dim = input_dim,
		latent_dim = args.latents, 
		encoder_z0 = encoder_z0, 
		decoder = decoder, 
		diffeq_solver = diffeq_solver, 
		z0_prior = z0_prior, 
		device = device,
		obsrv_std = obsrv_std,
		).to(device)

	return model


def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device):
	# dim related
	latent_dim = args.latents # ode output dimension
	rec_dim = args.rec_dims
	ode_dim = args.ode_dims #ode gcn dimension

	dim = args.latents
	if args.poisson:
		lambda_net = utils.create_net(dim, input_dim,
									  n_layers=1, n_units=args.units, nonlinear=nn.Tanh)

		# ODE function produces the gradient for latent state and for poisson rate
		ode_func_net = utils.create_net(dim * 2, args.latents * 2,
										n_layers=args.gen_layers, n_units=args.units, nonlinear=nn.Tanh)

		gen_ode_func = ODEFunc_w_Poisson(
			input_dim=input_dim,
			latent_dim=args.latents * 2,
			ode_func_net=ode_func_net,
			lambda_net=lambda_net,
			device=device).to(device)
	else:
		ode_func_net = utils.create_net(dim, args.latents,
										n_layers=args.gen_layers, n_units=args.units, nonlinear=nn.Tanh)

		gen_ode_func = ODEFunc(
			input_dim=input_dim,
			latent_dim=args.latents,
			ode_func_net=ode_func_net,
			device=device).to(device)


	enc_input_dim = int(input_dim) * 2  # we concatenate the mask
	gen_data_dim = input_dim

	z0_dim = args.latents
	if args.poisson:
		z0_dim += args.latents  # predict the initial poisson rate

	#encoder related
	if args.z0_encoder_lode == "odernn":
		ode_func_net = utils.create_net(rec_dim, rec_dim,
										n_layers=args.rec_layers, n_units=args.units, nonlinear=nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim=input_dim,
			latent_dim=rec_dim,
			ode_func_net=ode_func_net,
			device=device).to(device)

		z0_diffeq_solver = DiffeqSolver_lode(input_dim, rec_ode_func, "euler", args.latents,
										odeint_rtol=1e-3, odeint_atol=1e-4, device=device)

		encoder_z0 = Encoder_z0_ODE_RNN(rec_dim, input_dim, z0_diffeq_solver,
										z0_dim=args.latents, n_gru_units=args.gru_units, device=device).to(device)

	elif args.z0_encoder == "rnn":
		encoder_z0 = Encoder_z0_RNN(args.latents, input_dim,
									lstm_output_size=rec_dim, device=device).to(device)
	else:
		raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)


	#ODE related
	diffeq_solver = DiffeqSolver_lode(gen_data_dim, gen_ode_func, 'dopri5', args.latents,
								 odeint_rtol=1e-3, odeint_atol=1e-4, device=device)


    #Decoder related
	decoder = Decoder(latent_dim, input_dim).to(device)

	model = LatentODE(
		input_dim=input_dim,
		latent_dim=args.latents,
		encoder_z0=encoder_z0,
		decoder=decoder,
		diffeq_solver=diffeq_solver,
		z0_prior=z0_prior,
		device=device,
		obsrv_std=obsrv_std,
		use_poisson_proc=args.poisson,
		use_binary_classif=args.classif,
		linear_classifier=args.linear_classif,
		classif_per_tp=False,
		n_labels=1,
		train_classif_w_reconstr=(args.system == "physionet")
	).to(device)
	return model


def create_LatentODE_model_(args, input_dim, z0_prior, obsrv_std, device,
						   classif_per_tp=False, n_labels=1):
	dim = args.latents
	if args.poisson:
		lambda_net = utils.create_net(dim, input_dim,
									  n_layers=1, n_units=args.units, nonlinear=nn.Tanh)

		# ODE function produces the gradient for latent state and for poisson rate
		ode_func_net = utils.create_net(dim * 2, args.latents * 2,
										n_layers=args.gen_layers, n_units=args.units, nonlinear=nn.Tanh)

		gen_ode_func = ODEFunc_w_Poisson(
			input_dim=input_dim,
			latent_dim=args.latents * 2,
			ode_func_net=ode_func_net,
			lambda_net=lambda_net,
			device=device).to(device)
	else:
		dim = args.latents
		ode_func_net = utils.create_net(dim, args.latents,
										n_layers=args.gen_layers, n_units=args.units, nonlinear=nn.Tanh)

		gen_ode_func = ODEFunc(
			input_dim=input_dim,
			latent_dim=args.latents,
			ode_func_net=ode_func_net,
			device=device).to(device)

	z0_diffeq_solver = None
	n_rec_dims = args.rec_dims
	enc_input_dim = int(input_dim) * 2  # we concatenate the mask
	gen_data_dim = input_dim

	z0_dim = args.latents
	if args.poisson:
		z0_dim += args.latents  # predict the initial poisson rate

	if args.z0_encoder_lode == "odernn":
		ode_func_net = utils.create_net(n_rec_dims, n_rec_dims,
										n_layers=args.rec_layers, n_units=args.units, nonlinear=nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim=enc_input_dim,
			latent_dim=n_rec_dims,
			ode_func_net=ode_func_net,
			device=device).to(device)

		z0_diffeq_solver = DiffeqSolver_lode(enc_input_dim, rec_ode_func, "euler", args.latents,
										odeint_rtol=1e-3, odeint_atol=1e-4, device=device)

		encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver,
										z0_dim=z0_dim, n_gru_units=args.gru_units, device=device).to(device)

	elif args.z0_encoder_lode == "rnn":
		encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
									lstm_output_size=n_rec_dims, device=device).to(device)
	else:
		raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder_lode)

	decoder = Decoder(args.latents, gen_data_dim).to(device)

	diffeq_solver = DiffeqSolver_lode(gen_data_dim, gen_ode_func, 'dopri5', args.latents,
								 odeint_rtol=1e-3, odeint_atol=1e-4, device=device)

	model = LatentODE(
		input_dim=gen_data_dim,
		latent_dim=args.latents,
		encoder_z0=encoder_z0,
		decoder=decoder,
		diffeq_solver=diffeq_solver,
		z0_prior=z0_prior,
		device=device,
		obsrv_std=obsrv_std,
		use_poisson_proc=args.poisson,
		use_binary_classif=args.classif,
		linear_classifier=args.linear_classif,
		classif_per_tp=classif_per_tp,
		n_labels=n_labels,
		train_classif_w_reconstr=(args.system == "physionet")
	).to(device)

	return model





def create_CoupledODE_model(args, input_dim, z0_prior, obsrv_std, device):


	# dim related
	input_dim = input_dim
	output_dim = input_dim
	ode_hidden_dim = args.ode_dims
	rec_hidden_dim = args.rec_dims


	#ODE related
	if args.augment_dim > 0:  # Padding is done after the output of encoder. Encoder output dim is the expected ode_hidden_dim. True hidden dim is ode_hidden_dim + augment_dim
		ode_input_dim = ode_hidden_dim + args.augment_dim

	else:
		ode_input_dim = ode_hidden_dim


	rec_ouput_dim = ode_hidden_dim*2 # Need to split the vector into mean and variance (multiply by 2)


	#Encoder related

	encoder_z0 = GNN_cgode(in_dim=input_dim, n_hid=rec_hidden_dim, out_dim=rec_ouput_dim, n_heads=1,
						 n_layers=args.rec_layers, dropout=args.dropout, conv_name=args.z0_encoder,is_encoder=True, args = args)  # [b,n_ball,e]

	# ODE related
	# 1. Node ODE function
	node_ode_func_net = Node_GCN(in_dims = ode_input_dim,out_dims = ode_input_dim,num_atoms = args.n_balls,dropout=args.dropout)

	# 2. Edge ODE function
	w_node_to_edge_initial = nn.Linear(ode_input_dim * 2, ode_input_dim)  # h_ij = W([h_i||h_j])
	utils.init_network_weights(w_node_to_edge_initial)
	w_node_to_edge_initial = w_node_to_edge_initial.to(device)

	edge_ode_func_net = Edge_NRI(in_channels = ode_input_dim, w_node2edge = w_node_to_edge_initial, dropout=args.dropout,num_atoms = args.n_balls,device=device)

	# 3. Wrap Up ODE Function
	coupled_ode_func = CoupledODEFunc(
		node_ode_func_net=node_ode_func_net,
		edge_ode_func_net=edge_ode_func_net,
		device=device,
		num_atom = args.n_balls,dropout=args.dropout).to(device)



	diffeq_solver = DiffeqSolver_cgode(coupled_ode_func, args.solver, args=args,odeint_rtol=1e-2, odeint_atol=1e-2, device=device)

    #Decoder related
	decoder_node = Decoder_cdgoe(ode_hidden_dim, output_dim).to(device)
	decoder_edge = Decoder_cdgoe(ode_hidden_dim,1).to(device)

	model = CoupledODE(
		w_node_to_edge_initial = w_node_to_edge_initial,
		ode_hidden_dim = ode_hidden_dim,
		encoder_z0 = encoder_z0,
		decoder_node = decoder_node,
		decoder_edge = decoder_edge,
		diffeq_solver = diffeq_solver,
		z0_prior = z0_prior,
		device = device,
		obsrv_std = obsrv_std,
		n_balls = args.n_balls
		).to(device)

	#print_parameters(model)


	return model