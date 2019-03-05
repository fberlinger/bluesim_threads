"""Summary
"""
import math
import numpy as np
from queue import Queue
import time
import datetime

from events import HopCount, Ping, InfoInternal, LeaderElection
from eventcodes import (
    PING, HOMING, HOP_COUNT, INFO_EXTERNAL, INFO_INTERNAL, START_HOP_COUNT,
    START_LEADER_ELECTION, LEADER_ELECTION, MOVE
)


class Fish():
    """This class models each fish robot node in the collective from the fish'
    perspective.

    Each fish has an ID, communicates over the channel, and perceives its
    neighbors and takes actions accordingly. In taking actions, the fish can
    weight information from neighbors based on their distance. Different collective behaviors run different methods of this class. It can perceive and move according to its perceptual and dynamics model, and updates its behavior on every clock tick.

    Attributes:
        behavior (str): Behavior that fish follows
        body_length (int): Length of a BlueBot (130mm)
        caudal (int): Caudal fin control
        channel (Class): Communication channel
        clock (int): Local clock time
        clock_freq (float): Clock speed (Hz)
        clock_speed (float): Clock speed (s)
        d_center (int): Relative distance to center of perceived neighbors
        dorsal (int): Dorsal fin control
        dynamics (Class): Fish dynamics model
        fish_max_speed (int): Maximum forward speed of fish (old simulator)
        hop_count (int): Hop count variable
        hop_count_initiator (bool): Hop count started or not
        hop_distance (int): Hop distance to other fish
        id (int): ID number of fish
        info (str): Some information
        info_clock (int): Time stamp of the information, i.e., the clock
        info_hops (int): Number of hops until the information arrived
        initial_hop_count_clock (int): Hop count start time
        interaction (Class): Class for interactions between fish
        is_started (bool): True/false
        last_hop_count_clock (TYPE): Time since last hop count
        last_leader_election_clock (int): Time since last leader election
        leader_election_max_id (int): Highest ID
        lim_neighbors (int): Max. and min. desired number of neighbors
        messages (list): Messages between fish
        name (string): Name for logger file
        neighbor_weight (float): Gain that influeces decision making
        neighbors (set): Set of observed neighboring fish
        pect_l (int): Pectoral left fin control
        pect_r (int): Pectoral right fin control
        queue (TYPE): Message queue for messages from neighbors
        saw_hop_count (bool): True/false
        status (str): Behavioral status
        target_depth (int): Target depth for diving
        target_dist (int): Target distance for orbiting
        target_pos (int): Target position instructed by observer
        verbose (bool): Print statements on/off
    """

    def __init__(
        self,
        id,
        channel,
        interaction,
        dynamics,
        w_blindspot=50,
        r_blocking=65,
        target_dist=390,
        lim_neighbors=[0, math.inf],
        fish_max_speed=1,
        clock_freq=1,
        neighbor_weight=1.0,
        name='Unnamed',
        verbose=False
    ):
        """Create a new fish

        Arguments:
            id (TYPE): UUID of fish
            channel (Class): Communication channel
            interaction (Class): Class for interactions between fish
            dynamics (Class): Fish dynamics model
            target_dist (int, optional): target_distance to neighbors
            lim_neighbors (int, int): Lower and upper limit of neighbors each
                fish aims to be connected to.
                (default: {0, math.inf})
            fish_max_speed (float): Max speed of each fish. Defines by how
                much it can change its position in one simulation step.
                (default: {1})
            clock_freq (number): Behavior update rate in Hertz (default: {1})
            neighbor_weight (number): A weight based on distance that defines
                how much each of a fish's neighbor affects its next move.
                (default: {1.0})
            name (str): Unique name of the fish. (default: {'Unnamed'})
            verbose (bool): If `true` log out some stuff (default: {False})
        """

        self.id = id
        self.channel = channel
        self.interaction = interaction
        self.dynamics = dynamics
        self.w_blindspot = w_blindspot
        self.r_blocking = r_blocking
        self.target_dist = target_dist
        self.neighbor_weight = neighbor_weight
        self.lim_neighbors = lim_neighbors
        self.fish_max_speed = fish_max_speed
        self.clock_freq = clock_freq
        self.name = name
        self.verbose = verbose

        self.caudal = 0
        self.dorsal = 0
        self.pect_r = 0
        self.pect_l = 0
        self.target_depth = 0

        self.d_center = 0
        self.body_length = 130
        self.clock_speed = 1 / self.clock_freq
        self.clock = 0
        self.queue = Queue()
        self.target_pos = np.zeros((3,))
        self.is_started = False
        self.neighbors = set()

        self.status = None
        self.behavior = 'home'

        self.info = None  # Some information
        self.info_clock = 0  # Time stamp of the information, i.e., the clock
        self.info_hops = 0  # Number of hops until the information arrived
        self.last_hop_count_clock = -math.inf
        self.hop_count = 0
        self.hop_distance = 0
        self.hop_count_initiator = False
        self.initial_hop_count_clock = 0

        self.leader_election_max_id = -1
        self.last_leader_election_clock = -1

        now = datetime.datetime.now()

        # Stores messages to be sent out at the end of the clock cycle
        self.messages = []

        # Logger instance
        # with open('{}_{}.log'.format(self.name, self.id), 'w') as f:
        #     f.truncate()
        #     f.write('TIME  ::  #NEIGHBORS  ::  INFO  ::  ({})\n'.format(
        #         datetime.datetime.now())
        #     )

    def start(self):
        """Start the process

        This sets `is_started` to true and invokes `run()`.
        """
        self.is_started = True
        self.run()

    def stop(self):
        """Stop the process

        This sets `is_started` to false.
        """
        self.is_started = False

    def log(self, neighbors=set()):
        """Log current state
        """

        with open('{}_{}.log'.format(self.name, self.id), 'a+') as f:
            f.write(
                '{:05}    {:04}    {}    {}\n'.format(
                    self.clock,
                    len(neighbors),
                    self.info,
                    self.info_hops
                )
            )

    def run(self):
        """Run the process recursively

        This method simulates the fish and calls `eval` on every clock tick as
        long as the fish `is_started`.
        """

        while self.is_started:

            start_time = time.time()
            self.eval()
            time_elapsed = time.time() - start_time

            sleep_time = (self.clock_speed / 2) - time_elapsed

            # print(time_elapsed, sleep_time, self.clock_speed / 2)
            time.sleep(max(0, sleep_time))
            if sleep_time < 0 and self.verbose:
                print('Warning frequency too high or computer too slow')

            start_time = time.time()
            self.communicate()
            time_elapsed = time.time() - start_time

            sleep_time = (self.clock_speed / 2) - time_elapsed
            time.sleep(max(0, sleep_time))
            if sleep_time < 0 and self.verbose:
                print('Warning frequency too high or computer too slow')


    def move_handler(self, event):
        """Handle move events, i.e., update the target position.

        Arguments:
            event (Move): Event holding an x, y, and z target position
        """
        self.target_pos[0] = event.x
        self.target_pos[1] = event.y
        self.target_pos[2] = event.z

    def ping_handler(self, neighbors, rel_pos, event):
        """Handle ping events

        Arguments:
            neighbors {set} -- Set of active neighbors, i.e., nodes from which
                this fish received a ping event.
            rel_pos {dict} -- Dictionary of relative positions from this fish
                to the source of the ping event.
            event {Ping} -- The ping event instance
        """
        neighbors.add(event.source_id)

        # When the other fish is not perceived its relative position is [0,0]
        rel_pos[event.source_id] = self.interaction.perceive_pos(
            self.id, event.source_id
        )

        if self.verbose:
            print('Fish #{}: saw friend #{} at {}'.format(
                self.id, event.source_id, rel_pos[event.source_id]
            ))

    def homing_handler(self, event, pos):
        """Homing handler, i.e., make fish aggregated extremely

        Arguments:
            event {Homing} -- Homing event
            pos {np.array} -- Position of the homing event initialtor
        """
        self.info = 'signal_aircraft'  # Very bad practice. Needs to be fixed!
        self.info_clock = self.clock

        self.messages.append(
            (self, InfoInternal(self.id, self.clock, self.info))
        )

        # update behavior based on external event
        self.status = 'wait'
        self.target_pos = self.interaction.perceive_object(self.id, pos)

        if self.verbose:
            print('Fish #{} got external info {}'.format(
                self.id, event.message
            ))

    def info_ext_handler(self, event):
        """External information handler

        Always accept the external information and spread the news.

        Arguments:
            event {InfoExternal} -- InfoExternal event
        """
        self.info = event.message
        self.info_clock = self.clock

        self.messages.append(
            (self, InfoInternal(self.id, self.clock, self.info))
        )

        if self.verbose:
            print('Fish #{} got external info {}'.format(
                self.id, event.message
            ))

    def info_int_handler(self, event):
        """Internal information event handler.

        Only accept the information of the clock is higher than from the last
        information

        Arguments:
            event {InfoInternal} -- Internal information event instance
        """
        if self.info_clock >= event.clock:
            return

        self.info = event.message
        self.info_clock = event.clock
        self.info_hops = event.hops + 1

        self.messages.append((
            self,
            InfoInternal(self.id, self.info_clock, self.info, self.info_hops)
        ))

        if self.verbose:
            print('Fish #{} got info: {} from #{}'.format(
                self.id, event.message, event.source_id
            ))

    def hop_count_handler(self, event):
        """Hop count handler

        Initialize only of the last hop count event is 4 clocks old. Otherwise
        update the hop count and resend the new value only if its larger than
        the previous hop count value.

        Arguments:
            event {HopCount} -- Hop count event instance
        """
        # initialize
        if (self.clock - self.last_hop_count_clock) > 4:
            self.hop_count_initiator = False
            self.hop_distance = event.hops + 1
            self.hop_count = event.hops + 1
            self.messages.append((
                self,
                HopCount(self.id, self.info_clock, self.hop_count)
            ))

        else:
            # propagate value
            if self.hop_count < event.hops:
                self.hop_count = event.hops

                if not self.hop_count_initiator:
                    self.messages.append((
                        self,
                        HopCount(self.id, self.info_clock, self.hop_count)
                    ))

        self.last_hop_count_clock = self.clock

        if self.verbose:
            print('Fish #{} counts hops {} from #{}'.format(
                self.id, event.hop_count, event.source_id
            ))

    def start_hop_count_handler(self, event):
        """Hop count start handler

        Always accept a new start event for a hop count

        Arguments:
            event {StartHopCount} -- Hop count start event
        """
        self.last_hop_count_clock = self.clock
        self.hop_distance = 0
        self.hop_count = 0
        self.hop_count_initiator = True
        self.initial_hop_count_clock = self.clock

        self.messages.append((
            self,
            HopCount(self.id, self.info_clock, self.hop_count)
        ))

        if self.verbose:
            print('Fish #{} counts hops {} from #{}'.format(
                self.id, event.hop_count, event.source_id
            ))

    def leader_election_handler(self, event):
        """Leader election handler

        Arguments:
            event {LeaderElection} -- Leader election event instance
        """
        # This need to be adjusted in the future
        if (self.clock - self.last_leader_election_clock) < math.inf:
            new_max_id = max(event.max_id, self.id)
            # propagate value
            if self.leader_election_max_id < new_max_id:
                self.leader_election_max_id = new_max_id

                self.messages.append((
                    self,
                    LeaderElection(self.id, new_max_id)
                ))

        self.last_leader_election_clock = self.clock

    def weight_neighbor(self, rel_pos_to_neighbor): #xx obsolete with lj-pot?
        """Weight neighbors by the relative position to them

        Currently only returns a static value but this could be tweaked in the
        future to calculate a weighted center point.

        Arguments:
            rel_pos_to_neighbor {np.array} -- Relative position to a neighbor

        Returns:
            float -- Weight for this neighbor
        """
        return self.neighbor_weight

    def start_leader_election_handler(self, event):
        """Leader election start handler

        Always accept a new start event for a leader election

        Arguments:
            event {StartLeaderElection} -- Leader election start event
        """
        self.last_leader_election_clock = self.clock
        self.leader_election_max_id = self.id

        self.messages.append((
            self,
            LeaderElection(self.id, self.id)
        ))

    def comp_center(self, rel_pos):
        """Compute the (potentially weighted) centroid of the fish neighbors

        Arguments:
            rel_pos {dict} -- Dictionary of relative positions to the
                neighboring fish.

        Returns:
            np.array -- 3D centroid
        """
        center = np.zeros((3,))
        n = max(1, len(rel_pos))

        for key, value in rel_pos.items():
            weight = self.weight_neighbor(value)
            center += value * weight

        center /= n

        if self.verbose:
            print('Fish #{}: swarm centroid {}'.format(self.id, center))

        return center

    def lj_force(self, neighbors, rel_pos):
        """lj_force derives the Lennard-Jones potential and force based on the relative positions of all neighbors and the desired self.target_dist to neighbors. The force is a gain factor, attracting or repelling a fish from a neighbor. The center is a point in space toward which the fish will move, based on the sum of all weighted neighbor positions.

        Args:
            neighbors (set): Visible neighbors
            rel_pos (dict): Relative positions of visible neighbors

        Returns:
            np.array: Weighted 3D direction based on visible neighbors
        """
        if not neighbors:
            return np.zeros((3,))

        a = 12 # 12
        b = 6 # 6
        epsilon = 100 # depth of potential well, V_LJ(r_target) = epsilon
        gamma = 100 # force gain
        r_target = self.target_dist
        r_const = r_target + 1 * self.body_length

        center = np.zeros((3,))
        n = len(neighbors)

        for neighbor in neighbors:
            r = np.clip(np.linalg.norm(rel_pos[neighbor]), 0.001, r_const)
            f_lj = -gamma * epsilon /r * (a * (r_target / r)**a - 2 * b * (r_target / r)**b)
            center += f_lj * rel_pos[neighbor]

        center /= n

        return center

    def depth_ctrl(self, r_move_g):
        """Controls diving depth based on direction of desired move.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        pitch = np.arctan2(r_move_g[2], math.sqrt(r_move_g[0]**2 + r_move_g[1]**2)) * 180 / math.pi

        if pitch > 1:
            self.dorsal = 1
        elif pitch < -1:
            self.dorsal = 0

    def depth_waltz(self, r_move_g):
        """Controls diving depth in a pressure sensor fashion. Own depth is "measured", i.e. reveiled by the interaction. Depth control is then done based on a target depth coming from a desired goal location in the robot frame.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        depth = self.interaction.perceive_depth(self.id)

        if self.target_depth == 0:
            self.target_depth = depth + r_move_g[2] / 2

        if depth > self.target_depth:
            self.dorsal = 0
        else:
            self.dorsal = 1

    def home(self, r_move_g):
        """Homing behavior. Sets fin controls to move toward a desired goal location.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        caudal_range = 20 # abs(heading) below which caudal fin is switched on

        heading = np.arctan2(r_move_g[1], r_move_g[0]) * 180 / math.pi

        # target to the right
        if heading > 0:
            self.pect_l = min(1, 0.6 + abs(heading) / 180)
            self.pect_r = 0

            if heading < caudal_range:
                self.caudal = min(0.2, 0.1 + np.linalg.norm(r_move_g[0:2])/(8*self.body_length))
            else:
                self.caudal = 0

        # target to the left
        else:
            self.pect_r = min(1, 0.6 + abs(heading) / 180)
            self.pect_l = 0

            if heading > -caudal_range:
                self.caudal = min(0.2, 0.1 + np.linalg.norm(r_move_g[0:2])/(8*self.body_length))
            else:
                self.caudal = 0

    def collisions(self, r_move_g):
        """Local collision avoidance where r_move_g comes from a local Lennard-Jones potential.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        caudal_range = 20 # abs(heading) below which caudal fin is switched on

        heading = np.arctan2(r_move_g[1], r_move_g[0]) * 180 / math.pi

        # target to the right
        if heading > 0:
            self.pect_l = min(1, 0.6 + abs(heading) / 180)

            if heading < caudal_range:
                self.caudal = min(self.caudal+0.5, self.caudal+0.2 + np.linalg.norm(r_move_g[0:2])/(8*self.body_length))

        # target to the left
        else:
            self.pect_r += min(1, 0.6 + abs(heading) / 180)

            if heading > -caudal_range:
                self.caudal = min(self.caudal+0.5, self.caudal+0.2 + np.linalg.norm(r_move_g[0:2])/(8*self.body_length))

    def transition(self, r_move_g):
        """Transitions between homing and orbiting. Uses pectoral right fin to align tangentially with the orbit.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
        """
        self.caudal = 0
        self.pect_l = 0
        self.pect_r = 1

        heading = np.arctan2(r_move_g[1], r_move_g[0]) * 180 / math.pi

        if heading > 35:
            self.pect_r = 0
            self.behavior = 'orbit'

    def orbit(self, r_move_g, target_dist):
        """Orbits an object, e.g. two vertically stacked LEDs, at a predefined radius

        Uses four zones to control the orbit with pectoral and caudal fins. The problem is reduced to 2D and depth control is handled separately.
        Could make fin frequencies dependent on distance and heading, i.e., use proportianl control.

        Args:
            r_move_g (np.array): Relative position of desired goal location in robot frame.
            target_dist (int): Target orbiting radius, [mm]
        """
        dist = np.linalg.norm(r_move_g[0:2]) # 2D, ignoring z
        heading = np.arctan2(r_move_g[1], r_move_g[0]) * 180 / math.pi

        if dist > target_dist:
            if heading < 90:
                self.caudal = 0.45
                self.pect_l = 0
                self.pect_r = 0
            else:
                self.caudal = 0.3
                self.pect_l = 1
                self.pect_r = 0
        else:
            if heading < 90:
                self.caudal = 0.45
                self.pect_l = 0
                self.pect_r = 1
            else:
                self.caudal = 0.45
                self.pect_l = 0
                self.pect_r = 0

    def circling(self, neighbors, rel_pos):
        sensing_angle = 45 #deg

        if not neighbors:
            self.pect_l = 0
            self.pect_r = 0.5
            self.caudal = 0.1
            return
        
        someone = self.interaction.see_circlers(self.id, neighbors, rel_pos, sensing_angle)

        if someone:
            self.pect_r = 0
            self.pect_l = 0.5
            self.caudal = 0.1
        else:
            self.pect_l = 0
            self.pect_r = 0.5
            self.caudal = 0.1

    def move(self, neighbors, rel_pos):
        """Make a cohesion and target-driven move

        The move is determined by the relative position of the centroid and a
        target position and is limited by the maximum fish speed.

        Arguments:
            neighbors (TYPE): Description
            rel_pos (TYPE): Description
            neighbors {set} -- Set of active neighbors, i.e., other fish that
                responded to the most recent ping event.
            rel_pos {dict} -- Relative positions to all neighbors

        Returns:
            np.array -- Move direction as a 3D vector
        """

        # Get the centroid of the swarm
        #centroid_pos = np.zeros((3,))

        # Get the relative direction to the centroid of the swarm
        #centroid_pos = self.lj_force(neighbors, rel_pos)
        #self.d_center = np.linalg.norm(self.comp_center(rel_pos))

        #move = self.target_pos + centroid_pos

        # Global to Robot Transformation
        #r_T_g = self.interaction.rot_global_to_robot(self.id)
        #r_move_g = r_T_g @ move


        #self.depth_ctrl(r_move_g)
        #self.home(r_move_g)
        self.circling(neighbors, rel_pos)

        self.dynamics.update_ctrl(self.dorsal, self.caudal, self.pect_r, self.pect_l)
        final_move = self.dynamics.simulate_move(self.id)

        return final_move

    def update_behavior(self):
        """Update the fish behavior.

        This actively changes the cohesion strategy to either 'wait', i.e, do
        not care about any neighbors or 'signal_aircraft', i.e., aggregate with
        as many fish friends as possible.

        In robotics 'signal_aircraft' is a secret key word for robo-fish-nerds
        to gather in a secret lab until some robo fish finds a robo aircraft.
        """
        if self.status == 'wait':
            self.lim_neighbors = [0, math.inf]
        elif self.info == 'signal_aircraft':
            self.lim_neighbors = [math.inf, math.inf]

    def eval(self):
        """The fish evaluates its state

        Currently the fish checks all responses to previous pings and evaluates
        its relative position to all neighbors. Neighbors are other fish that
        received the ping element.
        """

        # Set of neighbors at this point. Will be reconstructed every time
        neighbors = set()
        rel_pos = {}

        self.saw_hop_count = False

        while not self.queue.empty():
            (event, pos) = self.queue.get()

            if event.opcode == PING:
                self.ping_handler(neighbors, rel_pos, event)

            if event.opcode == HOMING:
                self.homing_handler(event, pos)

            if event.opcode == START_HOP_COUNT:
                self.start_hop_count_handler(event)

            if event.opcode == HOP_COUNT:
                self.hop_count_handler(event)

            if event.opcode == INFO_EXTERNAL:
                self.info_ext_handler(event)

            if event.opcode == INFO_INTERNAL:
                self.info_int_handler(event)

            if event.opcode == START_LEADER_ELECTION:
                self.start_leader_election_handler(event)

            if event.opcode == LEADER_ELECTION:
                self.leader_election_handler(event)

            if event.opcode == MOVE:
                self.move_handler(event)

        if self.clock > 1:
            # Move around (or just stay where you are)
            self.interaction.blind_spot(self.id, neighbors, rel_pos)
            self.interaction.occlude(self.id, neighbors, rel_pos)
            self.interaction.move(self.id, self.move(neighbors, rel_pos))

        # Update behavior based on status and information - update behavior
        self.update_behavior()

        self.neighbors = neighbors

        # self.log(neighbors)
        self.clock += 1

    def communicate(self):
        """Broadcast all collected event messages.

        This method is called as part of the second clock cycle.
        """
        for message in self.messages:
            self.channel.transmit(*message)

        self.messages = []

        # Always send out a ping to other fish
        self.channel.transmit(self, Ping(self.id))
