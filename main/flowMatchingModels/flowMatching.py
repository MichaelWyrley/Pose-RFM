from abc import ABC, abstractmethod

class FlowMatching(ABC):
    @abstractmethod
    def sample_timestep(self, n):
        """
        Abstract method to sample a single timestep for n samples.
        This method should define how to generate a timestep value where t ~ U()
        
        :param n: Number of timestep samples to generate
        :return: A collection of sampled timesteps -> Tensor(n, 1)
        """
        pass

    @abstractmethod
    def gen_random_x(self, x_1):
        """
        Abstract method to generate a random variable x_0 based on the condition x_1.
        This method should define the generation process of x_0 when x_1 is given, likely sampling from standard normal distribution.
        
        :param x_1: The output type
        :return: A randomly generated x_0 based on x_1
        """
        pass

    @abstractmethod
    def conditional_flow(self, x_0, x_1, t):
        """
        Abstract method to compute the conditional flow between x_0 and x_1 at time t.
        The conditional flow (\phi_t(x)) where the flow is defined by the ordinary diferential equation from the paper "FLOW MATCHING FOR GENERATIVE MODELING"
        
        :param x_0: Initial state
        :param x_1: Final state
        :param t: Time at which the flow is computed
        :return: The computed flow at time t
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x_0, x_t, x_1, t, epsilon=0.00001):
        """
        Abstract method to compute the conditional vector field at a given point and time.
        The conditional vector field generates the conditional probability path based on the paper "FLOW MATCHING FOR GENERATIVE MODELING"
        
        :param x_0: Initial state
        :param x_t: conditional flow for time t
        :param x_1: Final state
        :param t: Time at which the vector field is computed
        :param epsilon: A small value used for numerical stability or calculations
        :return: The vector field at the specified conditions
        """
        pass

    @abstractmethod
    def train_step(self, x_1):
        """
        Abstract method to perform a single training step using x_1.
        This method should specify how the model updates or learns from each batch of data represented by x_1.
        
        :param x_1: Data used for the training step
        :return: Updates the model based on the training data
        """
        pass

    @abstractmethod
    def sample(self, n, timesteps=50, scale=1):
        """
        Abstract method to sample 'n' data points, over a defined number of timesteps.
        
        :param n: Number of samples to generate
        :param timesteps: Number of timesteps to simulate
        :param scale: Scaling factor for the timestep computation
        :return: A set of sampled data points
        """
        pass

    @abstractmethod
    def sample_full(self, n, timesteps=50, scale=1):
        """
        Abstract method to sample 'n' trajectories where each intermidiate timestep is also returned.
        Similar to sample(), but includes all timesteps computed
        
        :param n: Number of complete trajectories to generate
        :param timesteps: Number of timesteps per trajectory
        :param scale: Scaling factor for the simulation
        :return: A set of complete sampled trajectories 
        """
        pass