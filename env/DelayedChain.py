#%%
from typing import NamedTuple
import jax.numpy as jnp
import jax

#%%
class ChainParams(NamedTuple):
    """
    Settings retained across episodes within a single lifetime
    """

    chain_length: jax.Array

class ChainState(NamedTuple):
    """
    Information needed to advance the current episode
    """

    depth: jax.Array
    chain: jax.Array
    arrangement: jax.Array

class DelayedChain:
    """
    Environment for Delayed Chain (Appendix A.3: Delayed Chain MDP)
    """
    num_actions = 2
    max_chain_length = 30
    num_observations = 2 + 2 * max_chain_length

    def sample_params(self, key: jax.Array) -> ChainParams:
        chain_length = jax.random.randint(key,
            shape = (),
            minval = 5,
            maxval = self.max_chain_length + 1,
            )
        return ChainParams(chain_length = chain_length)

    def reset(self, key: jax.Array, params: ChainParams):
        initial_observation = jax.random.randint(
            key,
            shape = (),
            minval = 0,
            maxval = 2,
        )

        arrangement = jnp.where(
            initial_observation == 0,
            jnp.array([0, 1]),
            jnp.array([1, 0])
        )

        state = ChainState(
            depth = jnp.array(0),
            chain = jnp.array(0),
            arrangement = arrangement
        )

    def step(self,
             key: jax.Array,
             state: ChainState,
             action: jax.Array,
             params: ChainParams,
    ):
        chain = jnp.where(
            state.depth == 0,
            state.arrangement[action],
            state.chain,
        )

        depth = state.depth + 1

        next_state = ChainState(
            depth = depth,
            chain = chain,
            arrangement = state.arrangement,
        )

        done = depth >= params.chain_length

        terminal_reward = jnp.where(chain == 0, 1.0, -1.0)
        reward = jnp.where(done, terminal_reward, 0.0)

        observation = (2 + self.max_chain_length * chain + (depth - 1))

        return observation, next_state, reward, done