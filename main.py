from arxiv_tools import search_papers, extract_info

def main():
    # Test the search function
    topic = "machine learning"
    paper_ids = search_papers(topic, max_results=3)
    print(f"Found {len(paper_ids)} papers:")
    
    # Test the extract function with the first paper
    if paper_ids:
        first_paper_id = paper_ids[0]
        print(f"\nExtracting info for paper: {first_paper_id}")
        paper_info = extract_info(first_paper_id)
        print(paper_info)


if __name__ == "__main__":
    main()
