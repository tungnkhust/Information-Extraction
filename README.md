# Information Extraction
Recently, data is very helpful. From data, we can mining value and build very application.
With motivation, in this project we build a tool about information extraction filed. 
It can extract information in a text and detect relation between them.

---
# 1. Setup
### Install requirements
```commandline
pip install -r requirements.text
```
### Install docker
You can install docker by following:
- docker: [link](https://docs.docker.com/engine/install/ubuntu/)
- docker-compose: [link](https://docs.docker.com/compose/install/)

### Export python path:
```commandline
export PYTHONPATH=./
```
---
# 2.Run
### 2.1 Start neo4j database
Neo4j uses a property graph database model. A graph data structure consists of nodes (discrete objects) that can be connected by relationships.
We use neo4j to save iformation after extraction.
```commandline
sudo docker-compose up
```
## 2.2 Run tools

---
# 3. How to use with code
## 3.1 Neo4j Database
### Create a relation:
```python
from src.db_api.Neo4jDB import Neo4jDB
from src.schema.schema import Relation, Entity

db = Neo4jDB(
    uri="neo4j://localhost:7687",
    user="neo4j",
    password="160199"
)

s_e = Entity(entity="Per", value="Nguyen Van A")
t_e = Entity(entity="Loc", value="Ha Noi")
rel_type = "Live_In"
r = Relation(source_entity=s_e, target_entity=t_e, relation=rel_type)
db.create_relationship(r)
results = db.query_relation_entities(s_e)
for record in results:
    print(record)
```
After add relation to database, you can check in : http://localhost:7474/browser/ with clause:
```
MATCH (p:Per {value: "Nguyen Van A"}) return p
```

---
# 4. References
CoNLL04 data: [link](https://cogcomp.seas.upenn.edu/Data/ER/)

