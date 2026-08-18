"""Mock objects for external dependencies"""
import numpy as np

class MockITMLogic:
    """Mock implementation of itmlogic for testing when it's not available"""
    
    @staticmethod
    def qlrpfl(prop_dict):
        """Mock ITM qlrpfl function"""
        # Return a simple array-like result
        return np.array([100.0])  # 100 dB path loss
    
    @staticmethod
    def lrprop(distance_km, prop_dict):
        """Mock ITM lrprop function"""
        # Return a modified property dictionary
        result = prop_dict.copy()
        result["aref"] = 95.0  # 95 dB path loss
        return result

class MockSionnaRT:
    """Mock implementation of Sionna RT for testing when it's not available"""
    
    class Scene:
        def __init__(self):
            self.frequency = None
            self.tx_array = None
            self.rx_array = None
        
        def add(self, obj):
            pass
    
    class PlanarArray:
        def __init__(self, num_rows=1, num_cols=1, pattern="tr38901", polarization="V"):
            self.num_rows = num_rows
            self.num_cols = num_cols
            self.pattern = pattern
            self.polarization = polarization
    
    class Transmitter:
        def __init__(self, name="tx", position=None, display_radius=2):
            self.name = name
            self.position = position
            self.display_radius = display_radius
        
        def look_at(self, receiver):
            pass
    
    class Receiver:
        def __init__(self, name="rx", position=None, display_radius=2):
            self.name = name
            self.position = position
            self.display_radius = display_radius
    
    class PathSolver:
        def __init__(self):
            pass
        
        def __call__(self, scene, max_depth=3, los=True, specular_reflection=True,
                     diffuse_reflection=False, refraction=False):
            return MockPaths()
    
    @staticmethod
    def load_scene(scene_name):
        return MockSionnaRT.Scene()
    
    @staticmethod
    def scene():
        class MockScene:
            munich = "munich"
            simple_street_canyon = "simple_street_canyon"
        return MockScene()

class MockPaths:
    """Mock paths object for Sionna RT testing"""
    
    def cir(self, normalize_delays=True, out_type="numpy"):
        # Return mock channel impulse response
        if out_type == "numpy":
            # Return a 2D array with some energy
            a = np.array([[0.1+0.2j, 0.05+0.1j], [0.05+0.05j, 0.02+0.01j]])
            return a, None
        else:
            # Return list format
            a = [0.1+0.2j, 0.05+0.1j, 0.05+0.05j, 0.02+0.01j]
            return a, None

# Factory functions to get mocks
def get_mock_itm_logic():
    return MockITMLogic()

def get_mock_sionna_rt():
    return MockSionnaRT()