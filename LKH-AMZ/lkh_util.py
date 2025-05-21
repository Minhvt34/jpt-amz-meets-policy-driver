import lkh_solver
import numpy as np

def get_best_tour():
    """
    Extract the best tour from the LKH solver.
    
    Returns:
        A list representing the best tour found by the solver.
    """
    try:
        # Now use the binding function we added
        return lkh_solver.get_best_tour()
    except RuntimeError as e:
        raise RuntimeError(f"Failed to get best tour: {e}")

def get_dimension():
    """
    Get the dimension of the problem.
    
    Returns:
        The dimension (number of nodes) in the problem.
    """
    return lkh_solver.get_dimension()

def get_best_cost():
    """
    Get the cost of the best tour found.
    
    Returns:
        The cost of the best tour.
    """
    return lkh_solver.get_best_cost()

def write_tour_to_file(filename, tour, cost):
    """
    Write the tour to a file in the LKH format.
    
    Args:
        filename: Path to the output file
        tour: List representing the tour
        cost: Cost of the tour
    """
    with open(filename, 'w') as f:
        f.write("NAME : Best tour found\n")
        f.write(f"COMMENT : Cost = {cost}\n")
        f.write(f"DIMENSION : {len(tour) - 1}\n")
        f.write("TOUR_SECTION\n")
        
        for node in tour:
            f.write(f"{node}\n")
        
        f.write("-1\n")
        f.write("EOF\n")

def read_tour_from_file(filename):
    """
    Read a tour from a file in the LKH format.
    
    Args:
        filename: Path to the input file
        
    Returns:
        tuple: (tour, cost) where tour is a list of nodes and cost is the tour cost
    """
    tour = []
    cost = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("COMMENT : Cost ="):
                cost = int(line.split("=")[1].strip())
            elif line == "TOUR_SECTION":
                # Next lines are the tour
                tour_started = True
            elif line == "-1" or line == "EOF":
                # End of tour
                continue
            elif line[0].isdigit():
                # Tour node
                node = int(line.strip())
                if node != -1:  # Skip the terminator
                    tour.append(node)
    
    return tour, cost

def save_best_tour(filename):
    """
    Save the best tour found by the solver to a file.
    
    Args:
        filename: Path to the output file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        tour = get_best_tour()
        cost = get_best_cost()
        write_tour_to_file(filename, tour, cost)
        return True
    except Exception as e:
        print(f"Error saving tour: {e}")
        return False 