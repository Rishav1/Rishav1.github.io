---
layout: post
title:  "Monte Carlo based Markovian Decision Process AI model that learns how to play Super Mario Bros"
date:   2016-10-29
desc: "Monte Carlo based Markovian Decision Process AI model that learns how to play Super Mario Bros"
keywords: "mario, Markovian Decision Process, MDP, Monte Carlo, FCEUX, Lua"
categories: [Rein_learning]
tags: [mario, Markovian Decision Process, MDP, Monte Carlo, FCEUX, Lua]
icon: icon-html
---

This project was inspired by the Mario AI that went viral recently. The AI was designed to learn how to play Super Mario Bros game using a genetic algorithm, popularly known as Neural Evolution through Augmenting Topologies(NEAT). The logic behind NEAT is that by using a method to carefully selectively breed good performing Neural nets, and by mutating them slightly, the model can learn to optimize any black-box function effectively.
{: .text-justify}


<div align="center">
<iframe id="ytplayer" type="text/html" width="854" height="480" src="https://www.youtube.com/embed/qv6UVOQ0F44" frameborder="0" fs="0"></iframe>
<p>A NEAT model playing Super Mario Bros.</p>
</div>

This inspired me to try out some basic reinforcement learning models on my own. I read about Markovian Decision Process based RL models from a really good book, ["Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf){:target="_blank"}. Among the models presented in the book, I used a model based on Monte Carlo, which utilizes the positive properties of Dynamic Programming and Monte-Carlo models(also MDP based).
{: .text-justify}

For the purpose of creating an environment for the RL model, I used an open-sourced Nintendo Entertainment System(NES) simulator [FCEUX v2.2.2](http://www.fceux.com/web/home.html){:target="_blank"}, that supports [Lua](https://www.lua.org/){:target="_blank"} scripting to read/write RAM map and automate joy-pad inputs. A famous Lua library named [Torch](http://torch.ch/){:target="_blank"} was used for some mathematical computations. I reused some code of the MarI/O NEAT implementation by Seth Bling to read RAM map and setup the environment for my RL model. In this blog, I have tried to explain the concept of Monte Carlo learning model and my implementation of a modification to this model that learns to play Mario.
{: .text-justify}

Markov Decision process is a discrete time stochastic control process, that satisfies the condition that the state the system moves to, when the agent takes an action, depends only on the current state and not on the entire path of states it took to reach the current state. If the dynamics of the system, i.e. the transition probabilities __p\(s'\|s,a\)__ of reaching some state __s'__ when an action __a__ was taken at some state __s__, is completely know, we can find the probability of the system being in a state using the theory for Markov Decision Processes. If for each __s__ to __s'__ state transition, the agent is given a reward __r(s,s')__, and the agent wants to select actions for each state so that the net long-term &#955;-discounted rewards is maximized, then our problem reduces to finding an optimal __Policy__ (mapping of states to actions) that gives the best long-term rewards. An iterative method of finding the optimal __Policy__ for such an MDP is called Dynamic Programing, and is explained in Chapter 4 of [this book](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf){:target="_blank"}. In Dynamic Programing, we iteratively calculate the optimal __Value__ function, i.e. the function giving net expected returns for the optimal __Policy__ starting from the input state, and using this optimal value function, the optimal policy is calculated. 
{: .text-justify}

<p align="center"><img src="/static/assets/img/blog/Mario/MDP.png" alt="blog-image" width="640" height="480"></p>
<p align="center">An example MDP with three states corresponding actions with transition probabilities.</p>

<p align="center"><img src="/static/assets/img/blog/Mario/DP.png" alt="blog-image" width="640" height="480"></p>
<p align="center">Psuedocode of Dynamic Programming model for finding optimal Policy.</p>

The environment of Mario is a bit tricky. We don't completely know the transition probabilities of any action(Buttons to press on joy-pad) taken for any state(the grid of block world surrounding Mario). In other words, we don't know beforehand what next screen input would we see (and with what probability) if we decide to make Mario jump on the current screen input. To tackle this problem, we will use another RL method that works on similar grounds as Dynamic Programming, called Monte-Carlo method. The difference in this method is that instead of having using the transition probabilities(which is unavailable in our case) to find the __Value__ function, it runs episodes of environment simulations to approximate the optimal __Quality__ function (similar to __Value__ function), using which, it computes the optimal __Policy__. The benefit of this method is that we don't have to know the dynamics of the environment beforehand, just a way to extract data from this dynamic environment by running episodic simulations (which can be done in our case). A shortcoming of this method however is that it requires "Exploring Start", i.e. the freedom to start randomly from different states for different episodes, which is tough to implement(randomly initialize Mario from any place in the game) in our case.
{: .text-justify}

<p align="center"><img src="/static/assets/img/blog/Mario/MC-ES.png" alt="blog-image" width="640" height="480"></p>
<p align="center">Psuedocode of Monte Carlo model with exploring starts for finding optimal Policy.</p>

We can do away with the requirement of having exploring start by using an &#1013;-soft policy, i.e. regardless of whatever optimal policy we know of yet, their is always a small probability &#1013; that a non-optimal random action is taken at each state, which helps the algorithm to keep on exploring new optimal policies. The psuedocode of this algorithm is given below.
{: .text-justify}

<p align="center"><img src="/static/assets/img/blog/Mario/epsilon-MC.png" alt="blog-image" width="640" height="480"></p>
<p align="center">Psuedocode of &#1013;-soft policy Monte Carlo algorithm.</p>

The above model can be used for training our Mario AI. Let's first create an environment for the model, and then we will proceed on to the implementation of the model.
{: .text-justify}

```lua
player1 = 1
BoxRadius=6
InputSize = (BoxRadius*2+1)*(BoxRadius*2+1)
TIMEOUT=50

-- Import the requires libraries.
function set_imports()
  require("torch")
end

-- Function to set a particular level and world.
function set_level(world,level)
  memory.writebyte(0x75F,world)
  memory.writebyte(0x75C,level)
  memory.writebyte(0x760,level)
end

-- Function to set the global Mario location.
function getPositions()
  marioX = memory.readbyte(0x6D) * 0x100 + memory.readbyte( 0x86 )
  marioY = memory.readbyte(0x03B8)+16
  screenX = memory.readbyte(0x03AD)
  screenY = memory.readbyte(0x03B8)
end

-- Function to get a particular tile with respect to Mario.
function getTile(dx, dy)
  local x = marioX + dx + 8
  local y = marioY + dy - 16

  local page = math.floor(x/256)%2
  local subx = math.floor((x%256)/16)
  local suby = math.floor((y - 32)/16)
  local addr = 0x500 + page*13*16+suby*16+subx

  if suby >= 13 or suby < 0 then
          return 0
  end

  if memory.readbyte(addr) ~= 0 then
          return 1
  else
          return 0
  end
end

-- Function to get the enemy tile locations.
function getSprites()
  local sprites = {}
  for slot = 0,4 do
    local enemy = memory.readbyte( 0xF+slot)
    if enemy ~= 0 then
      local ex = memory.readbyte( 0x6E + tonumber(slot))* 0x100 + memory.readbyte( 0x87 + tonumber(slot))
      local ey = memory.readbyte( 0xCF + slot) + 24
      sprites[#sprites+1] = {["x"]=ex,["y"]=ey}
    end
  end
  return sprites
end

-- Function to create a Box around Mario and return an input vector of the same. Free Free tile is 0, enemy tile is -1 and block tile is 1.
function getInputs()
  getPositions()
  sprites = getSprites()
  local inputs = {}

  for dy=-BoxRadius*16,BoxRadius*16,16 do
    for dx=-BoxRadius*16,BoxRadius*16,16 do
      inputs[#inputs+1] = 0

      tile = getTile(dx, dy)
      if tile == 1 and marioY+dy < 0x1B0 then
        inputs[#inputs] = 1
      end

      for i = 1,#sprites do
        distx = math.abs(sprites[i]["x"] - (marioX+dx))
        disty = math.abs(sprites[i]["y"] - (marioY+dy))
        if distx <= 8 and disty <= 8 then
          inputs[#inputs] = -1
        end
      end
    end
  end
  return inputs
end

-- Function returns true if Mario is in air.
function isInAir()
  if memory.readbyte(0x001D) ~= 0x00 then
    return true
  else 
    return false 
  end
end

-- Function returns true of Mario dies.
function isDead ()
  if memory.readbyte(0x000E) == 0x0B or memory.readbyte(0x000E) == 0x06 then
    return true;
  else
    return false;
  end
end


-- Function to create a save state at the start of a selected world and level.
function set_state ()
  while(memory.readbyte(0x0772)~=0x03 or memory.readbyte(0x0770)~=0x01) do
    set_level(01,01)
    emu.frameadvance()
  end
  level_start = savestate.object(1)
  savestate.save(level_start)
end

``` 

We need to call __set_state\(\)__ before the start of the execution of the model, to create a loadable save point at the beginning of the game level. The function __getInputs\(\)__ is used to get an input vector of size __InputSize__ that describes the state of the process. Each state of the MDP process is determined using the tiles surrounding Mario. The function cuts a box of edge size 2*__BoxRadius__ around Mario and returns it in a vectored form. A tile in the box is represented by 0 if it is an empty space, 1 if it is a block space(like ground or brick-tiles) and -1 if their is an enemy on the tile. Now lets look at the Monte Carlo implementation.
{: .text-justify}

```lua
function set_values()
  Returns={}
  Policy={}
  TIMEOUT=50
  TIMEOUT_AIR=10
  EPSILON=0.05
end

function getHashReturns(Inputs, Action)
  return getHashPolicy(Inputs).."->"..Action
end

function getHashPolicy(Inputs)
  local I_value=""
  for i=1,#Inputs do
    I_value=Inputs[i]..I_value
  end
  return I_value
end

function getReturnsForPair(Inputs,Action)
  if Returns[getHashReturns(Inputs,Action)]==nil then
    return 0
  else
    return Returns[getHashReturns(Inputs,Action)][1]
  end
end
```
I created a hash function that converts the __Input__ into a string and then hashes it. This is for quick retrial of policies based on the hashing string. The __getHashReturns\(Inputs, Action\)__ concatenates the __Action__ as a string to the __Input__ hash. The function __getReturnsForPair\(Inputs,Action\)__ returns the __Returns__ of the __Input__, __Action__ pair.
{: .text-justify}

```lua

function getButtonsForAction(Action)
  local action_hash=Action
  b_str=""
  local buttons = {["A"]=false,["B"]=false,["up"]=false,["down"]=false,["left"]=false,["right"]=false}
  if action_hash%2==1 then
    buttons["right"]=true
    buttons["left"]=false
    b_str=b_str.."right, "
  else
    buttons["right"]=false
    buttons["left"]=true
    b_str=b_str.."left, "
  end
  action_hash=math.floor(action_hash/2)
  if action_hash%2==1 then
    buttons["up"]=false
  else
    buttons["up"]=true
    b_str=b_str.."up, "
  end
  action_hash=math.floor(action_hash/2)
  if action_hash%2==1 then
    buttons["B"]=true
    b_str=b_str.."B, "
  else
    buttons["B"]=false
  end
  action_hash=math.floor(action_hash/2)
  if action_hash%2==1 then
    buttons["A"]=true
    b_str=b_str.."A, "
  else
    buttons["A"]=false
  end
  action_hash=math.floor(action_hash/2)
  if not action_hash==0 then
    eum.message("error, set button for hash didn't receive correct value\n")
  end
  emu.message(b_str)
  return buttons
end
```

The __Action__ parameter is an integer value, whose binary representation gives the combination of buttons to press. For simplicity and ease of training,  I have added some constraints that at any time either Left or Right button must be pressed and Down button is never pressed. The function returns a dictionary of Buttons with their corresponding boolean values indicating if the button is pressed.
{: .text-justify}

```lua
function generateAction(Inputs)
  local action_taken
  g_str=""
  if Policy[getHashPolicy(Inputs)]==nil then
    action_taken=torch.multinomial(torch.ones(1,16),1)
    action_taken=action_taken[1][1]
    g_str=g_str..",new "
  else
    g_str=g_str..",found "
    local x,y=torch.max(torch.Tensor(Policy[getHashPolicy(Inputs)]),1)
    action_taken=torch.multinomial(torch.Tensor(Policy[getHashPolicy(Inputs)]),1)
    action_taken=action_taken[1]
    if action_taken==y[1] then
      g_str=g_str..",taken "
    else
      g_str=g_str..",not taken "
    end
  end
  action_taken= action_taken-1
  return action_taken
end
```
The above function generates an action based on the __Input__ state and the current __Policy__. If the state is being seen for the first time, a random action is taken. If the state is not new, an action is chosen based on the __Policy__. Since the Policy is an &#1013;-soft Monte Carlo, their is always a non-zero probability of choosing a non optimal action for any __Input__ state.
{: .text-justify}

```lua
-- The reward function.
function getReward()
  local time_left=memory.readbyte(0x07F8)*100 + memory.readbyte(0x07F9)*10 + memory.readbyte(0x07FA)
  local dist = marioX
  return time_left*dist
end
```
The above function is the reward function being called at the end of each episode to get the reward. The reward is the product of distance traveled and the time remaining. We want the AI to reach the end of the level as quick as possible.

```lua
function run_episode()
  savestate.load(level_start)
  maxMario=nil
  local memory={}
  local state = {}
  local action_for_state=nil
  
  while true do
    state=getInputs()
    action_for_state=generateAction(state)
    joypad.set(player1,getButtonsForAction(action_for_state))
    emu.frameadvance()
    memory[#memory+1]={state,action_for_state}
    
    if isDead() then
      emu.message("Dead")
      break
    end

    local timeout=TIMEOUT
    local timeout_air=TIMEOUT_AIR
    local long_idle=false

    while(getHashPolicy(getInputs())==getHashPolicy(state)) do
      joypad.set(player1,getButtonsForAction(action_for_state))
      emu.frameadvance()
      if timeout==0 then
        long_idle=true
        break
      end
      timeout = timeout-1
    end

    while(isInAir()) do
      joypad.set(player1,getButtonsForAction(action_for_state))
      emu.frameadvance()
      if timeout_air==0 then
        break
      end
      timeout_air = timeout_air-1
    end

    if long_idle then
      emu.message("Killed Due to timeout")
      break
    end

  end
  return memory
end
```

The above function runs an episode of the game. It reloads to the start point (which is saved while initializing the environment). The function repetitively takes the input state of the environment and takes an action based on __Policy__. I added a timeout to end episode if it stays idle for too long. Also I disabled taking an action while in air to reduce the size of the __memory__ list. The function keeps track of all the __Input__, __Action__ pairs seen in the form of a __memory__ list, and returns this list at the end of the episode.
{: .text-justify}

```lua
function start_training()
  local Policy_number=1
  while true do
  
    run_memory=run_episode()
    run_reward=getReward()
    print("Policy is "..Policy_number)
    print("Run is "..i)
    print("Reward->"..run_reward)
    print("memory size->"..#run_memory)
    local new_states=0
    for i=1,#run_memory do
      local hashed_pair=getHashReturns(run_memory[i][1],run_memory[i][2])
      if Returns[hashed_pair]==nil then
        new_states = new_states + 1
        Returns[hashed_pair]={run_reward,1}
      else
        Returns[hashed_pair][1]=(Returns[hashed_pair][1]+run_reward)/(Returns[hashed_pair][2] + 1)
        Returns[hashed_pair][2]=Returns[hashed_pair][2] + 1
      end
    end
    print("Number of New States is "..new_states)
  

    for i=1,#run_memory do
      
      local max=-1
      local optimal_action=-1
      if Policy[getHashPolicy(run_memory[i][1])]==nil then
        Policy[getHashPolicy(run_memory[i][1])]={}
      end
      
      for j=0,15 do 
        if getReturnsForPair(run_memory[i][1],j) > max then
          max=getReturnsForPair(run_memory[i][1],j)
          optimal_action=j
        end
      end
            
      for j=0,11 do
        if j==optimal_action then
          Policy[getHashPolicy(run_memory[i][1])][j+1]= 1 - EPSILON +EPSILON/11
        else
          Policy[getHashPolicy(run_memory[i][1])][j+1]= EPSILON/11
        end
      end
    end
    Policy_number=Policy_number+1
  end
end

```

The above function trains the AI by running episodes and updating the __Policy__. It gets the __memory__ of the episode and then iterates through all the state-action pairs, updating the __Returns__ dictionary. Then it iterates through the memory again to update the __Policy__ for the states seen in the last episode. Since the policy is an Epsilon soft policy, we have to keep a minimum non-zero probability for all actions.
{: .text-justify}

```lua
function init ()
  set_imports()
  set_state()
  set_values()
  emu.speedmode("maximum")
end
```

The __init\(\)__ function is called before __start_training\(\)__ function to set the imports, the environment, to initialize the dictionary and global values, and to set the emulator speed. Now all that is left is to call the functions and observe the training of AI.
{: .text-justify}

```lua
init()
start_training()
```

Here's the first 1.5 minutes of the training of the model. We can clearly see how the AI improves as it discovers better paths.

<div align="center">
<iframe id="ytplayer" type="text/html" width="640" height="480" src="https://www.youtube.com/embed/eKQA3UmfcVM" frameborder="0" fs="0"></iframe>
<p>First 1.5 minutes of &#1013;-soft Monte Carlo training run.</p>
</div>

The model was able to complete Level 1 of World 1 after around 3 hours of training. Though this model is not as strong as NEAT model, which trains the AI to be able to play any level once trained, this model is simple enough for beginners to try and get impressive results.

The code for this project is open sourced, and can be found [here](https://github.com/Rishav1/Super-Mario-Bros-RL-AI){:target="_blank"}.