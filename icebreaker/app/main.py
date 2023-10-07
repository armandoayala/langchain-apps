from .third_parties import linkedin
from .agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


def run_lab():
    linkedin_data = linkedin_lookup_agent(name="Armando Ayala")
    print(linkedin_data)
