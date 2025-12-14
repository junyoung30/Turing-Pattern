from abc import ABC, abstractmethod


class PatternSolver(ABC):
    
    @abstractmethod
    def solve(self, params, seed):
        """
        Args:
            params (dict): simulation parameters
            seed (int): random seed
            
        Returns:
            generated pattern (np.ndarray)
        """
        pass