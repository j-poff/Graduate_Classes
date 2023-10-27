#!/usr/bin/env python3
class LowHeadDam:
    def __init__(self, id_name: str, stream_type: str, downstream_slope: float, width: float, depth: float, number_of_fatalities: int) -> None:
       
        # Setting instance variable to a shorter name
        self.id: str = id_name
        self.shape: str = stream_type
        self.slope: float = downstream_slope
        self.width: float = width
        self.depth: float = depth
        self.fatal: int = number_of_fatalities
        
    def give_name(self)-> str:
        name: str = self.id
        return name  

    def Mannings_equation(self)-> float:
        flow: float = 0.0
        n: float = .03
        if self.shape == "main channel, clean":
            n: float = .03
        if self.shape == "main channel, weedy and slugish":
            n: float = .06
        if self.shape == "mountain channel, few boulders":
            n: float = .04
        area = self.depth * self.width
        flow = (1.49/n) * area * (area / (2*self.width + self.depth))**(2/3) * self.slope **(1/2)
        flow = round(flow,1)
        return flow      

    def flat_fatalities(self) -> bool:
        fatal_: bool = False
        flat_: bool = False
        if self.fatal != 0:
            fatal_ = True
        if self.slope > 0.0005:
            flat_ = True
            
        return flat_, fatal_
    
    def flow_fatality_comparison(self) -> str:
        q: float = self.Mannings_equation()
        fatal: int = self.fatal
        report: str = "FLow is " + str(q) + " cfs with " + str(fatal) + " fatalities"
    
        return report

#create list to put objects into
lowheaddam_list = []
#define first one
oregon1: LowHeadDam = LowHeadDam('oregon1', "main channel, weedy and slugish", 0.0001, 15,2,0)
lowheaddam_list.append(oregon1)
#append other items
lowheaddam_list.append(LowHeadDam('california2', "main channel, clean", 0.0006, 30,6,3))
lowheaddam_list.append(LowHeadDam('washington1', "mountain channel, few boulders", 0.001, 4,0.5,2))
lowheaddam_list.append(LowHeadDam('utah1', "main channel, weedy and slugish", 0.00003, 10,1,5))

print("What is the estimated flow in cfs at these low head dams?")
for dam in lowheaddam_list:
    q = dam.Mannings_equation()
    print(str(dam.give_name()) + " has a flow of " + str(q) + "cfs")
    
print("Next, how many of these dams have had fatalities, and what percentage of them are flat?")
fatal_dams_counter: int = 0
flat_and_fatal_counter: int = 0
for dam in lowheaddam_list:
    flat, fatal = dam.flat_fatalities()
    if fatal == True:
        fatal_dams_counter += 1
    if fatal == True and flat == True:
        flat_and_fatal_counter += 1
print("Dams with fatalities: " + str(fatal_dams_counter))
percentage_flat: float = round(100*flat_and_fatal_counter/fatal_dams_counter,1)
print("Out of those, " + str(percentage_flat) + "% are flat")

for dam in lowheaddam_list:
    print(dam.flow_fatality_comparison())


