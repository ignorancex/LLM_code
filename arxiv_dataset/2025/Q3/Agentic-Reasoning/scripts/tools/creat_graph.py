from nano_graphrag import GraphRAG, QueryParam
from langchain.tools import BaseTool, StructuredTool, tool
import json

class MindMap:
    def __init__(self, ini_content = "", working_dir="./local_mem" ):
        """
        Initialize the graph with a specified working directory.
        """
        self.working_dir = working_dir
        self.graph_func = GraphRAG(working_dir=self.working_dir)
                # Read the content from book.txt and insert into the graph
        content = ini_content
        self.graph_func.insert(content)
    
    def process_community_report(self, json_path="local_mem/kv_store_community_reports.json") -> str:
        """
        Read and process the community report JSON, returning the combined report string.
        """
        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Collect all report strings from each community
        all_reports = []
        for community_id, community in data.items():
            report_string = community.get("report_string", "")
            all_reports.append(f"Snippet {community_id}:\n{report_string}\n")
        
        # Combine all reports
        combined_reports = "\n".join(all_reports)
        return combined_reports

    def graph_retrieval(self, query: str) -> None:
        """
        Insert content from './book.txt' into the graph index,
        then demonstrate both a global and a local query.
        """
        # # Perform a global graphrag search and print the result
        # print("Global search result:")
        # print(self.graph_func.query(query))
        
        # Perform a local graphrag search and print the result
        print("\nLocal graph search result:")
        res = self.graph_func.query(query, 
                                    param=QueryParam(mode="local"))
        print(res)

        return res
    
    def graph_query(self, query: str) -> str:
        """
        Retrieve community reports by processing the local JSON store.
        """
        combined_report = self.process_community_report()
        print("\ncombined community report is:", combined_report)

        query = f"Answer the question:{query}\n\n based on the information:\n\n{combined_report}"

        return self.process_community_report()

    def __call__(self, query):
        """
        query the mind map knowledge graph and return the result
        """
        return self.graph_retrieval(query)

# Example usage:
if __name__ == "__main__":
    # Create an instance of CreateGraph
    graph_manager = MindMap(working_dir="./local_mem")
    
    # Call graph_query which reads from './book.txt', inserts into GraphRAG, and prints query outputs
    graph_manager.graph_query("your query here")
    
    # Retrieve combined community report and print it
    combined_report = graph_manager.graph_retrieval("dummy query")
    print("\nCombined Community Report:")
    print(combined_report)