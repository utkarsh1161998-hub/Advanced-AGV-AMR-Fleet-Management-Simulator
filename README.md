import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================================================
# 1. PATHFINDING ENGINE (A* Algorithm)
# =====================================================================
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distance

def a_star(grid, start, goal):
    """Standard A* pathfinding on a 2D binary grid (0=free, 1=obstacle)."""
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
 
    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1] # Return reversed path

        close_set.add(current)
        for i, j in [(-1,0), (1,0), (0,-1), (0,1)]: # 4-directional movement
            neighbor = current[0] + i, current[1] + j
            
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue # Obstacle
            else:
                continue # Out of bounds
                
            tentative_g_score = gscore[current] + 1
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [x[1] for x in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return [] # Path not found

# =====================================================================
# 2. AMR / AGV AGENT DEFINITION
# =====================================================================
class Robot:
    def __init__(self, robot_id, start_pos):
        self.id = robot_id
        self.pos = start_pos          # Current (x, y)
        self.path = []                # List of coordinates to follow
        self.status = "IDLE"          # IDLE, MOVING_TO_PICK, MOVING_TO_DROP
        self.assigned_task = None
        self.color = np.random.rand(3,)

    def update_position(self, reserved_positions):
        """Moves one step along its allocated path if the cell is clear."""
        if not self.path:
            return
        
        next_step = self.path[0]
        
        # Simple Traffic Control / Conflict Resolution:
        # If the next node is already reserved by another robot this tick, hold position (wait)
        if next_step in reserved_positions:
            self.status = "BLOCKED / WAITING"
            return 
            
        # Move forward
        self.pos = self.path.pop(0)
        reserved_positions.add(self.pos)
        
        if not self.path:
            if self.status == "MOVING_TO_PICK":
                # Arrived at pickup, switch pathing to final delivery drop-off
                self.status = "MOVING_TO_DROP"
                self.path = self.assigned_task['drop_path']
            elif self.status == "MOVING_TO_DROP":
                # Task finished successfully
                self.status = "IDLE"
                self.assigned_task = None

# =====================================================================
# 3. FLEET MANAGEMENT SYSTEM (FMS) ENGINE
# =====================================================================
class FleetManager:
    def __init__(self, grid_width, grid_height):
        # Create a warehouse map with some static obstacles (shelves)
        self.grid = np.zeros((grid_width, grid_height))
        self.add_warehouse_shelves()
        
        self.robots = []
        self.task_queue = []

    def add_warehouse_shelves(self):
        """Generates mock warehouse grid layouts."""
        # Generating layout blocks representing shelf aisles
        for x in [3, 4, 7, 8, 11, 12, 15, 16]:
            for y in range(2, 14):
                self.grid[x, y] = 1

    def add_robot(self, start_pos):
        r_id = len(self.robots) + 1
        self.robots.append(Robot(r_id, start_pos))

    def submit_task(self, pickup, dropoff):
        """Adds a transport order requests to the FMS queue."""
        self.task_queue.append({'pickup': pickup, 'dropoff': dropoff})

    def dispatch_tasks(self):
        """Dispatches idle AGVs using a Greedy Closest-Distance Matcher."""
        idle_robots = [r for r in self.robots if r.status == "IDLE"]
        
        while self.task_queue and idle_robots:
            task = self.task_queue.pop(0)
            
            # Find closest idle robot to the pickup position
            best_robot = None
            min_dist = float('inf')
            
            for robot in idle_robots:
                dist = heuristic(robot.pos, task['pickup'])
                if dist < min_dist:
                    min_dist = dist
                    best_robot = robot
            
            if best_robot:
                # Calculate routing
                to_pickup_path = a_star(self.grid, best_robot.pos, task['pickup'])
                to_drop_path = a_star(self.grid, task['pickup'], task['dropoff'])
                
                if to_pickup_path and to_drop_path:
                    best_robot.assigned_task = {
                        'pickup': task['pickup'],
                        'dropoff': task['dropoff'],
                        'drop_path': to_drop_path[1:] # Drop starting node to prevent stuttering
                    }
                    best_robot.path = to_pickup_path[1:]
                    best_robot.status = "MOVING_TO_PICK"
                    idle_robots.remove(best_robot)
                else:
                    # If routing failed (e.g. trapped), put task back in queue
                    self.task_queue.insert(0, task)
                    break

    def step(self):
        """Executes one discrete simulation clock tick."""
        self.dispatch_tasks()
        
        reserved_positions = set()
        # Lock current robot spaces to handle immediate collisions
        for r in self.robots:
            reserved_positions.add(r.pos)
            
        for r in self.robots:
            # Clear previous spot from immediate check so the robot can move forward
            if r.pos in reserved_positions and r.path:
                reserved_positions.remove(r.pos)
            r.update_position(reserved_positions)

# =====================================================================
# 4. LIVE SIMULATION RUNNER & VISUALIZATION
# =====================================================================
if __name__ == "__main__":
    # Initialize Fleet Manager (Grid size 20x16)
    fms = FleetManager(20, 16)
    
    # Deploy Fleet of 4 AMRs at different starting stations
    fms.add_robot((0, 0))
    fms.add_robot((0, 15))
    fms.add_robot((19, 0))
    fms.add_robot((19, 15))
    
    # Generate random logistics orders (Pick/Drop coordinates)
    # Ensure points don't land directly inside walls/shelves
    valid_coords = np.argwhere(fms.grid == 0)
    for _ in range(8):
        p_idx = np.random.choice(len(valid_coords))
        d_idx = np.random.choice(len(valid_coords))
        pickup = tuple(valid_coords[p_idx])
        drop = tuple(valid_coords[d_idx])
        fms.submit_task(pickup, drop)

    # Setup Matplotlib Animation Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        fms.step()
        
        # Draw warehouse layout (0 = white/floor, 1 = grey/shelves)
        ax.imshow(fms.grid.T, cmap='binary', origin='lower', alpha=0.3)
        
        # Plot active robots and their intended path vectors
        for robot in fms.robots:
            # Draw remaining paths
            if robot.path:
                path_pts = np.array(robot.path)
                ax.plot(path_pts[:, 0], path_pts[:, 1], color=robot.color, linestyle=':', alpha=0.7)
                
            # Draw Robot position
            ax.scatter(robot.pos[0], robot.pos[1], color=robot.color, s=250, edgecolors='black', zorder=5)
            ax.text(robot.pos[0] - 0.3, robot.pos[1] - 0.2, f"R{robot.id}", color='white', weight='bold', fontsize=9, zorder=6)
            
            # Highlight target jobs if actively working
            if robot.assigned_task:
                p = robot.assigned_task['pickup']
                d = robot.assigned_task['dropoff']
                ax.scatter(p[0], p[1], marker='^', color=robot.color, s=100, label=f"Pick R{robot.id}" if frame==0 else "")
                ax.scatter(d[0], d[1], marker='v', color=robot.color, s=100, label=f"Drop R{robot.id}" if frame==0 else "")
                
        ax.set_title(f"Advanced AMR Fleet Management Simulation System — Tick: {frame}")
        ax.set_xticks(range(20))
        ax.set_yticks(range(16))
        ax.grid(True, which='both', color='lightgrey', linestyle='-', linewidth=0.5)
        
    ani = FuncAnimation(fig, update, frames=60, repeat=False, interval=400)
    plt.show()
