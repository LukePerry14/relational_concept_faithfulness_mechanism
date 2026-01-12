
class Orchestrator:
    
    def __init__(self, sampler, encoder, params):
        
        self.sampler = sampler
        self.encoder = encoder
        
        self.params = params