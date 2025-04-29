<p align="center">
    <img src="https://img.shields.io/badge/contributions-welcome!-green" alt="Contributions welcome!"/>
    <img src="https://img.shields.io/github/last-commit/omarstfa/pyfte?color=blue">
</p>

# Python for Fault Tree Extraction (PyFTE)
This is a tool for extracting fault tree models from system data. The fault logs are simulated using Monte carlo of failure probabilites for basic events.

## Getting Started

:information_source: *Tested with Python 3.12.3*

```bash
git clone https://github.com/omarstfa/PyFTE.git  # 1. Clone repository
pip install -r requirements.txt  # 2. Install requirements
python3 examples/PV_example.py  # 3. Run PV-system example
```

# Case Study Model (PV system example)

The PV system layout:
![image1](https://github.com/user-attachments/assets/26e142ad-b603-464c-b6ba-e0b6eb1af141)

The original fault tree model of the system:
![image2](https://github.com/user-attachments/assets/fc8e2b42-a820-411e-96ec-5e6aa4ea79dc)

The input code for the original fault tree model:
```bash
topEvent = Event('Top Event')

or0 = Gate('or', parent=topEvent)
intermediateEvent1 = Event('Intermediate Event 1', parent=or0)
intermediateEvent2 = Event('Intermediate Event 2', parent=or0)

or1 = Gate('OR', parent=intermediateEvent1)
basicEvent1 = Event('Basic Event 1', parent=or1)
basicEvent2 = Event('Basic Event 2', parent=or1)
intermediateEvent3 = Event('Intermediate Event 3', parent=or1)

or2 = Gate('or', parent=intermediateEvent3)
intermediateEvent4 = Event('Intermediate Event 4', parent=or2)
intermediateEvent5 = Event('Intermediate Event 5', parent=or2)

or3 = Gate('OR', parent=intermediateEvent4)
basicEvent5 = Event('Basic Event 5', parent=or3)
basicEvent6 = Event('Basic Event 6', parent=or3)
basicEvent7 = Event('Basic Event 7', parent=or3)
basicEvent8 = Event('Basic Event 8', parent=or3)
basicEvent9 = Event('Basic Event 9', parent=or3)
basicEvent10 = Event('Basic Event 10', parent=or3)
basicEvent11 = Event('Basic Event 11', parent=or3)
basicEvent12 = Event('Basic Event 12', parent=or3)
basicEvent13 = Event('Basic Event 13', parent=or3)
basicEvent14 = Event('Basic Event 14', parent=or3)
basicEvent15 = Event('Basic Event 15', parent=or3)
basicEvent16 = Event('Basic Event 16', parent=or3)

and1 = Gate('AND', parent=intermediateEvent5)
basicEvent3 = Event('Basic Event 3', parent=and1)
basicEvent4 = Event('Basic Event 4', parent=and1)

or4 = Gate('OR', parent=intermediateEvent2)
basicEvent16 = Event('Basic Event 17', parent=or4)
basicEvent17 = Event('Basic Event 18', parent=or4)
```

# Output

```bash
Identifying Cut Sets (first 5 shown):
Cut Set 1: ['BE18']
Cut Set 2: ['BE17']
Cut Set 3: ['BE17', 'BE18']
Cut Set 4: ['BE18', 'BE4']
Cut Set 5: ['BE17', 'BE4']

Minimal Cut Sets: 
['BE18'], ['BE17'], ['BE16'], ['BE15'], ['BE14'], ['BE13'], ['BE12'], ['BE11'], ['BE10'], ['BE9'], ['BE8'], ['BE7'], ['BE6'], ['BE5'], ['BE2'], ['BE1'], ['BE3', 'BE4']

Extracted Fault Tree Boolean Expression:  TE = BE17 + BE16 + BE2 + BE1 + BE15·BE3 + BE14·BE3 + BE13·BE3 + BE12·BE3 + BE11·BE3 + BE10·BE3 + BE3·BE9 + BE3·BE8 + BE3·BE7 + BE3·BE6 + BE3·BE5 + BE3·BE4

Constructed Truth Table (sample):
R   BE1  BE2  BE3  BE4  BE5  BE6  BE7  ...    BE13  BE14  BE15  BE16  BE17  BE18  TE
0    0    0    0    0    0    0    0   ...     0     0     0     0     0     0     0
1    0    0    0    0    0    0    0   ...     0     0     0     0     0     1     1
2    0    0    0    0    0    0    0   ...     0     0     0     0     1     0     1
3    0    0    0    0    0    0    0   ...     0     0     0     0     1     1     1
4    0    0    0    1    0    0    0   ...     0     0     0     0     0     0     0
[5 rows x 19 columns]

Truth table generated and saved from Boolean expression.
[Validation Successful]: Truth Tables Match!
The constructed fault tree produces an identical truth table to the original.
```

# Usage & Attribution

For any questions please contact omar.mostafa@kit.edu
