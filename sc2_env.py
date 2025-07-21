from pysc2.env import sc2_env
from pysc2.lib import actions, features
import numpy as np

class SC2Env:
    def __init__(self, map_name):
        self.env = sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=None,
            visualize=False
        )
        
    def reset(self):
        self.current_obs = self.env.reset()[0]
        return self._process_state(self.current_obs)
    
    def step(self, action):
        # self.current_obs = self.env.step([self._process_action(action)])[0]
        action_id, args = self._process_action(action)
        self.current_obs = self.env.step([actions.FunctionCall(action_id, args)])[0]
        state = self._process_state(self.current_obs)
        reward = self.current_obs.reward
        done = self.current_obs.last()
        return state, reward, done, args
    
    def get_available_actions(self):
        """Get list of available actions from current observation."""
        return self.current_obs.observation["available_actions"]
    
    def _process_state(self, obs):
        screen = obs.observation.feature_screen
        # Select relevant features (you can modify this based on your needs)
        processed_screen = np.stack([
            screen[features.SCREEN_FEATURES.player_relative.index],
            screen[features.SCREEN_FEATURES.unit_density.index],
            screen[features.SCREEN_FEATURES.unit_type.index]
        ])
        return processed_screen
    
    def _process_action(self, action_id):
        """Convert network output to SC2 action with appropriate arguments."""
        try:
            # Convert and validate action_id
            action_id = int(action_id)
            # print(f"\n")
            # print(f"Processing action_id: {action_id}")
            
            # Check if action_id is valid
            if not (0 <= action_id < len(actions.FUNCTIONS)):
                print(f"Invalid action_id: {action_id}, defaulting to no_op")
                return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

            # Get action info first
            try:
                action_info = actions.FUNCTIONS[action_id]
                # print(f"Action info: {action_info.name}, Args required: {action_info.args}")
            except Exception as e:
                print(f"Error getting action info for {action_id}: {e}")
                return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

            # Check if action is available
            if action_id not in self.current_obs.observation["available_actions"]:
                print(f"Action {action_id} not in available actions: {self.current_obs.observation['available_actions']}")
                return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

            # Rest of your action processing code...
            action_info = actions.FUNCTIONS[action_id]
            # args = []

            # print(f"action id == point: {action_id in [actions.FUNCTIONS.select_point.id]}")
            # Basic actions with no args
            # if action_id in [actions.FUNCTIONS.no_op.id, 
            #                actions.FUNCTIONS.Stop_quick.id]:
                
            #             #    actions.FUNCTIONS.Smart_screen.id]:
            #     # args = []
            #     # print("No args needed")
            if action_id in [actions.FUNCTIONS.move_camera.id]:
                args = [[0, 0]] # [x, y]
            # Screen coordinate actions
            elif action_id in [
                # Movement and targeting
                actions.FUNCTIONS.Move_screen.id,
                actions.FUNCTIONS.Attack_screen.id,
                actions.FUNCTIONS.Patrol_screen.id,
                actions.FUNCTIONS.Smart_screen.id,
                actions.FUNCTIONS.Scan_Move_screen.id,
                # Building placement
                actions.FUNCTIONS.Build_Armory_screen.id,
                actions.FUNCTIONS.Build_Barracks_screen.id,
                actions.FUNCTIONS.Build_Bunker_screen.id,
                actions.FUNCTIONS.Build_CommandCenter_screen.id,
                actions.FUNCTIONS.Build_EngineeringBay_screen.id,
                actions.FUNCTIONS.Build_Factory_screen.id,
                actions.FUNCTIONS.Build_FusionCore_screen.id,
                actions.FUNCTIONS.Build_GhostAcademy_screen.id,
                actions.FUNCTIONS.Build_MissileTurret_screen.id,
                actions.FUNCTIONS.Build_Refinery_screen.id,
                actions.FUNCTIONS.Build_SensorTower_screen.id,
                actions.FUNCTIONS.Build_Starport_screen.id,
                actions.FUNCTIONS.Build_SupplyDepot_screen.id,
                # Unit abilities
                actions.FUNCTIONS.Effect_Heal_screen.id,
                actions.FUNCTIONS.Effect_Repair_screen.id,
                actions.FUNCTIONS.Harvest_Gather_screen.id,
                actions.FUNCTIONS.Rally_Building_screen.id,
                actions.FUNCTIONS.Rally_Units_screen.id,
                actions.FUNCTIONS.Rally_Workers_screen.id
            ]:
                args = [[0], self._get_beacon_position()]  # [queued, [x, y]]

            # Minimap coordinate actions
            elif action_id in [
                actions.FUNCTIONS.Attack_minimap.id,
                actions.FUNCTIONS.Move_minimap.id,
                actions.FUNCTIONS.Patrol_minimap.id,
                actions.FUNCTIONS.Rally_Building_minimap.id,
                actions.FUNCTIONS.Rally_Units_minimap.id,
                actions.FUNCTIONS.Rally_Workers_minimap.id,
                actions.FUNCTIONS.Smart_minimap.id
            ]:
                args = [[0], [32, 32]]  # [queued, [x, y]]

            # Selection actions
            elif action_id in [
                actions.FUNCTIONS.select_point.id,
                actions.FUNCTIONS.select_unit.id
            ]:
                # print(f"testttttttttt") # Debugging
                args = [[0], self._get_beacon_position()]  # [select_type, [x, y]]
                # print(f" Selecting unit at {self._get_beacon_position()}")  # Debugging
            elif action_id == actions.FUNCTIONS.select_rect.id:
                pos = self._get_beacon_position()
                args = [[0], pos, [pos[0] + 2, pos[1] + 2]]  # [select_add, [x1, y1], [x2, y2]]
                # print(f"Selecting rectangle from {pos} to {[pos[0] + 2, pos[1] + 2]}")  # Debugging
            elif action_id in [
                actions.FUNCTIONS.select_idle_worker.id,
                actions.FUNCTIONS.select_army.id,
                actions.FUNCTIONS.select_warp_gates.id
            ]:
                args = [[0]]  # [select_type]

            # Control group actions
            elif action_id in [
                actions.FUNCTIONS.select_control_group.id,
                # actions.FUNCTIONS.control_group.id
            ]:
                args = [[0], [0]]  # [control_group_act, control_group_id]

            # Quick training actions (no location needed)
            elif action_id in [
                actions.FUNCTIONS.Train_Marine_quick.id,
                actions.FUNCTIONS.Train_Marauder_quick.id,
                actions.FUNCTIONS.Train_Reaper_quick.id,
                actions.FUNCTIONS.Train_Ghost_quick.id,
                actions.FUNCTIONS.Train_Hellion_quick.id,
                actions.FUNCTIONS.Train_Hellbat_quick.id,
                actions.FUNCTIONS.Train_SiegeTank_quick.id,
                actions.FUNCTIONS.Train_Thor_quick.id,
                actions.FUNCTIONS.Train_Medivac_quick.id,
                actions.FUNCTIONS.Train_Liberator_quick.id,
                actions.FUNCTIONS.Train_Raven_quick.id,
                actions.FUNCTIONS.Train_Banshee_quick.id,
                actions.FUNCTIONS.Train_Battlecruiser_quick.id,
                actions.FUNCTIONS.Train_SCV_quick.id
            ]:
                args = [[1]]  # [queued]

            # Research actions
            elif action_id in [
                actions.FUNCTIONS.Research_CombatShield_quick.id,
                actions.FUNCTIONS.Research_ConcussiveShells_quick.id,
                actions.FUNCTIONS.Research_Stimpack_quick.id,
                actions.FUNCTIONS.Research_TerranInfantryArmor_quick.id,
                actions.FUNCTIONS.Research_TerranInfantryWeapons_quick.id,
                # actions.FUNCTIONS.Research_TerranVehicleArmor_quick.id,
                actions.FUNCTIONS.Research_TerranVehicleWeapons_quick.id,
                # actions.FUNCTIONS.Research_TerranShipArmor_quick.id,
                actions.FUNCTIONS.Research_TerranShipWeapons_quick.id
            ]:
                args = [[0]]  # [queued]

            # Ability/Effect actions
            elif action_id in [
                actions.FUNCTIONS.Effect_Stim_quick.id,
                actions.FUNCTIONS.Effect_EMP_screen.id,
                actions.FUNCTIONS.Effect_Heal_screen.id,
                actions.FUNCTIONS.Effect_Repair_screen.id,
                # actions.FUNCTIONS.Effect_Repair_SCV_quick.id,
                # actions.FUNCTIONS.Effect_Repair_MULE_quick.id,
                actions.FUNCTIONS.Effect_Salvage_quick.id
                # actions.FUNCTIONS.Effect_Scan_Move_screen.id,
                # actions.FUNCTIONS.Effect_Yamato_screen.id
            ]:
                if '_screen' in action_info.name:
                    args = [[0], self._get_beacon_position()]  # [queued, [x, y]]
                else:
                    args = [[0]]  # [queued]

            # Default handling for any unspecified actions
            else:
                args = [[0] for _ in action_info.args]  # Default args for each parameter

            # print action_id and args
            # print(f"Action ID: {action_id}, Args: {args}")  # Debugging
            # return actions.FunctionCall(action_id, args)
            # print(f"action info: {action_info.name},   action_id: {action_id}, args: {args}")
            return action_id, args
            
        except Exception as e:
            print(f"Exception in _process_action:")
            print(f"- Action ID: {action_id}")
            print(f"- Error: {str(e)}")
            print(f"- Available actions: {self.current_obs.observation['available_actions']}")
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    def _get_unit_position(self, unit_type):
        """Helper method to find position of a specific unit type."""
        unit_layer = self.current_obs.observation.feature_screen[features.SCREEN_FEATURES.unit_type.index]
        y, x = (unit_layer == unit_type).nonzero()
        if len(y) > 0:
            return [x[0], y[0]]
        return [42, 42]  # Default to center if unit not found

    def _get_beacon_position(self):
        """Get the position of the beacon or default to center."""
        beacon_layer = self.current_obs.observation.feature_screen[features.SCREEN_FEATURES.player_relative.index]
        beacon_y, beacon_x = (beacon_layer == features.PlayerRelative.NEUTRAL).nonzero()
        if len(beacon_y) > 0:
            # Using beacon position without arbitrary offsets for better precision
            return [beacon_x[0]+5, beacon_y[0]+5]
        return [42, 42]  # Default to center if beacon not found
    
    def close(self):
        """Close the environment properly."""
        if hasattr(self, 'env'):
            self.env.close()
