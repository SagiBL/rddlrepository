# files to update in venv

## Description
The two python files in this folder should replace the files in the venv in order to match our implemetation. 

## files location in venv
venv/Lib/site-packages/pyRDDLGym/core

## functions added in env:
    def set_state(self, state):
        sampler = self.sampler
        sampler.set_state(state)
        self.state = state
        
## functions added in simulator:
    def update_subs(self, setstate):
        rddl = self.rddl
        for key, value in setstate.items():
            if key.startswith('flow-on-link'):
                link_part = key.split('___')[1]
                link, time = link_part.split('__t')
                link_idx = int(link[1:])
                time_idx = int(time)
                self.subs['flow-on-link'][link_idx, time_idx] = value
            if key.startswith('q'):
                link1 = key.split('__l')[1]
                link2 = key.split('__l')[2]
                link1_idx = int(link1)
                link2_idx = int(link2)
                self.subs['q'][link1_idx, link2_idx] = value
            if key.startswith('Nc'):
                link_part = key.split('___l')[1]
                link_idx = int(link_part)
                self.subs['Nc'][link_idx] = value
            if key.startswith('virtual-q'):
                link_part = key.split('___l')[1]
                link_idx = int(link_part)
                self.subs['virtual-q'][link_idx] = value
            if key.startswith('signal') and not key.startswith('signal-t'):
                link_part = key.split('___i')[1]
                link_idx = int(link_part)
                self.subs['signal'][link_idx] = value
            if key.startswith('signal-t'):
                link_part = key.split('___i')[1]
                link_idx = int(link_part)
                self.subs['signal-t'][link_idx] = value
        for (state, next_state) in rddl.next_state.items():
            self.subs[next_state] = self.subs[state]

    def set_state(self, setstate: Args) -> Args:
        rddl = self.rddl
        keep_tensors = self.keep_tensors
        self.update_subs(setstate)
        subs = self.subs
      
    
