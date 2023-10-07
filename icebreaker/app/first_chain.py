from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from .third_parties import linkedin
from .agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


information = """
Nelson Rolihlahla Mandela (/mænˈdɛlə/; Xhosa: [xolíɬaɬa mandɛ̂ːla]; nacido Rolihlahla Mandela (Mvezo, Provincia Cabo Oriental; 18 de julio de 1918-Johannesburgo, Gauteng; 5 de diciembre de 2013) fue un abogado, activista contra el apartheid, político y filántropo sudafricano que presidió el gobierno de su país de 1994 a 1999. Fue el primer mandatario de raza negra que encabezó el poder ejecutivo, y el primero en resultar elegido por sufragio universal en su país. Su gobierno se dedicó a desmontar la estructura social y política heredada del apartheid a través del combate del racismo institucionalizado, la pobreza, la desigualdad social y la promoción de la reconciliación social. Como nacionalista africano y marxista, presidió el Congreso Nacional Africano (CNA) entre 1991 y 1997, y a nivel internacional fue secretario general del Movimiento de Países No Alineados entre 1998 y 2002.

Originario del pueblo xhosa y parte de la casa real tembu, Mandela estudió Derecho en la Universidad de Fort Hare y la Universidad de Witwatersrand. Cuando residía en Johannesburgo, se involucró en la política anticolonialista, por lo que se unió a las filas del Congreso Nacional Africano, y luego fundó su Liga Juvenil. Tras la llegada al poder del Partido Nacional en 1948, ganó protagonismo durante la Campaña del Desafío de 1952 y fue elegido presidente regional del Congreso Nacional Africano en la provincia de Transvaal. Presidió el Congreso Popular de 1955. En su ejercicio como abogado, fue varias veces arrestado por actividades sediciosas y, como parte de la directiva del CNA, fue procesado en el Juicio por Traición desde 1956 hasta 1961. Influenciado por el marxismo, entró en secreto al Partido Comunista Sudafricano (SACP) y fue parte de su comité central. Pese a que estaba a favor de las protestas no violentas, en asociación con la SACP fundó y comandó la organización guerrillera Umkhonto we Sizwe (MK) o «La Lanza de la Nación» en 1961.1​ En 1962 fue arrestado y acusado de conspiración para derrocar al gobierno, por lo que fue sentenciado a prisión perpetua durante el Proceso de Rivonia.
"""


def run_app():
    print("Langchain!")

    linkedin_data = linkedin_lookup_agent(name="Armando Ayala")

    summary_template = """
         given the information {information} about a person from I want you to create:
         1. short summary
         2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = linkedin.scrape_linkedin_profile(linkedin_data)
    print(chain.run(information=linkedin_data))
