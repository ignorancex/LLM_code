import haiku as hk
from functools import partial
import jax
import jax.numpy as jnp
import os
import pickle
import optax
import tqdm
import wandb
from util import recursive_items, optim_adadelta, grad_norm
from jax.experimental.host_callback import id_print

class LSVAETrainer:
    def __init__(self, build_lsvae):
        self.build_lsvae = build_lsvae

    def train(self, config, test_data, train_data, val_extra, model_extra,
                    seed=jax.random.PRNGKey(42), load=None):
        seed_seq = hk.PRNGSequence(seed)
        def f():
            @partial(hk.jit, static_argnums=(2,3))
            def run_batch(i, batch, do_extra, is_training):
                lsvae = self.build_lsvae(i)
                inputs = batch['inputs']
                meas = dict(batch)
                del meas['inputs']

                lsvae_res = lsvae.compute(hk.next_rng_key(), 
                                meas, inputs, is_training, [])
                extra = model_extra(hk.next_rng_key(), batch, lsvae, lsvae_res) \
                            if do_extra else None
                return lsvae_res, extra

            def init(batch):
                _ = run_batch(0, batch, True, True)
            return init, run_batch

        f = hk.multi_transform_with_state(f)
        print('initializing')
        params, state = f.init(next(seed_seq), next(test_data))
        print('initialized')
        run_batch = f.apply

        #opt = optax.adam(optax.exponential_decay(config.learning_rate, 10000, 0.5))
        opt = optax.adamw(config.learning_rate)
        # opt = optim_adadelta(0.1)
        #opt = optim_adadelta(config.learning_rate)
        opt_state = opt.init(params)

        @jax.jit
        def update(i, params, state, rng, opt_state, batch):
            def apply(params):
                (lsvae_res, _), new_state = run_batch(params, state, rng, i, batch, False, True)
                return lsvae_res['stats']['loss'], (lsvae_res['stats'], new_state)
            grads, (res, new_state) = jax.grad(apply, has_aux=True)(params)

            #grads = jax.tree_map(lambda x: jnp.clip(x, -500, 500), grads)
            gn = jax.tree_map(lambda x: grad_norm(x, '2'), grads)

            updates, opt_state = opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state, opt_state, res, gn

        @jax.jit
        def evaluate(i, params, state, rng, batch):
            (res, extra), _ = run_batch(params, state, rng, i, batch, True, False)
            return res, extra

        if load is None:
            print('Pretraining decoder')
            t = tqdm.tqdm(range(500))
            pretrain_rng = hk.PRNGSequence(next(seed_seq))

            for step in t:
                batch = next(train_data)
                rng = next(pretrain_rng)
                new_params, new_state, opt_state, res, _ = \
                    update(1000, params, state, rng, opt_state, batch)
                filter = lambda m, n, p: m.startswith('decoder')
                train_new, _ = hk.data_structures.partition(filter, new_params)
                _, other = hk.data_structures.partition(filter, params)
                # train_new_state, _ = hk.data_structures.partition(filter, new_state)
                # _, other_state = hk.data_structures.partition(filter, state)
                # state = hk.data_structures.merge(train_new_state, other_state)
                state = new_state

                params = hk.data_structures.merge(train_new, other)
                # update the new_params, new_state only corresponding to the decoder
            print('Pretraining finished')

            # re-initialize optimizer state
            opt_state = opt.init(params)
        else:
            pass

        t = tqdm.tqdm(range(config.iterations))

        val_rng = hk.PRNGSequence(424242)
        train_rng = hk.PRNGSequence(4242)

        loss = None
        val_loss = None

        for step in t:
            t_sk = next(train_rng)
            batch = next(train_data)
            if step % config.save_interval == 0 and step != 0:
                save_checkpoint(f'cp-{step}', params, opt_state, state, config)
            if step % config.val_interval == 0:
                if step == 0:
                    print('Running eval')
                v_sk = next(val_rng)
                val_batch = next(test_data)

                val_res, val_extra_res = evaluate(10000, params, state, v_sk, val_batch)
                train_res, train_extra_res = evaluate(10000, params, state, t_sk, batch)

                val_loss = val_res['stats']['loss']

                val_stats = {
                    f'val/{stat}': v for stat, v in recursive_items(val_res['stats'])
                }
                val_extras = {
                    f'val/{stat}': v for stat, v in val_extra(batch, val_res, val_extra_res).items()
                }
                train_extras = {
                    f'val/{stat}': v for stat, v in val_extra(batch, train_res, train_extra_res).items()
                }
                wandb.log(val_stats, commit=False)
                wandb.log(val_extras, commit=False)
                wandb.log(train_extras, commit=False)
                t.set_postfix(loss=loss, val_loss=val_loss)
                if step == 0:
                    print("Eval finished")
            params, state, opt_state, res, gn = \
                update(step, params, state, t_sk, opt_state, batch)
            stats = {
                f'train/{stat}': v for stat, v in recursive_items(res)
            }
            stats.update({
                f'train_grads/{var}': v for var, v in recursive_items(gn)
            })
            # print out covariance log-likelihoods
            # stats.update({
            #     f'train/L_{i}': v for i, v in enumerate(params['encoder']['L'])
            # })
            loss=res['loss']
            t.set_postfix(loss=loss, val_loss=val_loss)
            wandb.log(stats)
        save_checkpoint('final', params, opt_state, state, config)


def save_checkpoint(name, params, opt_state, state, config):
    os.makedirs(f'checkpoints/{wandb.run.id}/{name}', exist_ok=True)
    pickle.dump(params, open(f"checkpoints/{wandb.run.id}/{name}/params.pk", "wb"))
    pickle.dump(opt_state, open(f"checkpoints/{wandb.run.id}/{name}/opt_state.pk", "wb"))
    pickle.dump(state, open(f"checkpoints/{wandb.run.id}/{name}/state.pk", "wb"))
    # convert the config to a dict
    config_dict = {k: v for k, v in config.items()}
    pickle.dump(config_dict, open(f"checkpoints/{wandb.run.id}/{name}/config.pk", "wb"))