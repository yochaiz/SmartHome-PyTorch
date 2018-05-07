import json
from datetime import datetime, timedelta
from torch import Tensor
from random import randint


# This policy represents my typical week behavior as my house.
# The policy is built from 7 days, 24 hours a day
# List of objects I try to predicts:
## Lights:
### my room - 3
### kitchen - 1
### toilets - 1
### bathroom - 1
### living room - 2
### hallway - 1
### entrance door - 1
#
## Boiler - 1 (we will assume winter time at the moment, i.e. every time I shower it requires boiler)
#
# Input structure:
# [Weekday (0-6), Hour (0-23), Minute (0-60), Room light1, Room light2, Room light3,
#  Kitchen light, Toilets light, Bathroom light, Living room light1, Living room light2,
#  Hallway light, Entrance light, Boiler]
#
# Output structure:
# [Room light1, Room light2, Room light3, Kitchen light, Toilets light,
#  Bathroom light, Living room light1, Living room light2, Hallway light,
#  Entrance light, Boiler]

# Policy object structure:
# Policy is a dictionary with key per device.
## Each key contains list of time dictionaries {'days':[] , 'times':[]}
### each time dictionary contains 2 keys:
#### days: array of days the device should be ON.
#### times: array of time interval during these days that the device should be ON.
##### each time interval in times is a tuple (startTime , endTime)


class Policy:
    # action is a discrete vector
    # 0: DO NOT change device state
    # 1: change device state
    actionPossibleValues = (0, 1)

    # init state date data: (resolution name, (minValue, maxValue))
    stateDateData = [('Weekday', (0, 6)), ('Hour', (0, 23)), ('Minute', (0, 59))]
    stateDevicesStartIdx = len(stateDateData)

    # set values for state time normalization
    # timeNormalizationValues = Tensor([6, 24, 60])
    # assert (len(stateDateTitle) == len(timeNormalizationValues))

    # stateDim - state vector dimensions
    # actionDim - action vector dimensions.
    # action vector is a discrete vector with values from {0,1}
    # TODO: work with IntTensor most of time rather than FloatTensor ???
    def __init__(self, fname, tensorType):
        self.tensorType = tensorType
        # self.timeNormalizationValues.type(self.tensorType)

        self.policyJSON = self.loadPolicyFromJSON(fname)
        self.numOfDevices = len(self.policyJSON["Devices"])

        self.stateDim = self.stateDevicesStartIdx + self.numOfDevices
        self.actionDim = self.numOfDevices

        # init possible actions
        self.possibleActions, self.nActions = self.__buildPossibleActions()

    def __buildPossibleActions(self):
        nActions = pow(len(self.actionPossibleValues), self.actionDim)
        actions = [[v] for v in self.actionPossibleValues]
        for _ in range(self.actionDim - 1):
            newPerm = []
            for p in actions:
                for v in self.actionPossibleValues:
                    newPerm.append(p + [v])
            actions = newPerm

        # convert list to torch tensor
        actions = Tensor(actions).type(self.tensorType)

        return actions, nActions

    def minTimeUnit(self):
        return timedelta(minutes=1)

    # dateTensor - a tensor represents date data
    # resolutionList - name of time resolution of each value, i.e. weekday, hour, minute ,second
    # returns a datetime obj of date tensor data
    def dateTensorToObj(self, dateTensor, resolutionList):
        # TODO: make it more generic, connect between strings here and in stateDateData
        weekDay = int(dateTensor[resolutionList.index('Weekday')]) if 'Weekday' in resolutionList else 0
        hour = int(dateTensor[resolutionList.index('Hour')]) if 'Hour' in resolutionList else 0
        minutes = int(dateTensor[resolutionList.index('Minute')]) if 'Minute' in resolutionList else 0
        seconds = int(dateTensor[resolutionList.index('Second')]) if 'Second' in resolutionList else 0
        # 05/02/2018 is Monday which is (weekday == 0)
        # it synchronizes between month day and weekday, i.e. same value for both
        dateObj = datetime(2018, 2, 5 + weekDay, hour, minutes, seconds)

        return dateObj

    # builds new state given current state and action
    # curState & action are tensors
    def buildNewState(self, curState, action):
        assert (curState.type() == self.tensorType)
        assert (action.type() == self.tensorType)

        newState = curState.clone()
        newState[self.stateDevicesStartIdx:] += action
        return newState

    def generateRandomDate(self):
        dateTensor = Tensor(len(self.stateDateData)).fill_(0).type(self.tensorType)
        resolutionList = []
        for i, (resName, (minVal, maxVal)) in enumerate(self.stateDateData):
            dateTensor[i] = Tensor(1).random_(minVal, maxVal + 1)[0]
            resolutionList.append(resName)

        # create datetime object
        dateObj = self.dateTensorToObj(dateTensor, resolutionList)

        return dateTensor, dateObj

    def generateRandomState(self):
        state = Tensor(self.stateDim).fill_(0).type(self.tensorType)
        # set random date
        state[:self.stateDevicesStartIdx], dateObj = self.generateRandomDate()
        # set random devices state
        state[self.stateDevicesStartIdx:] = Tensor(self.numOfDevices) \
            .random_(min(self.actionPossibleValues), max(self.actionPossibleValues) + 1).type(self.tensorType)
        # TODO: generate random state by using expected state ???

        return state, dateObj

    # Builds the expected state at given date
    # TODO: do we need the time part in the returned tensor ???
    def buildExpectedState(self, dateTensor, dateObj):
        assert (dateTensor.type() == self.tensorType)
        state = Tensor(self.stateDim).fill_(0).type(self.tensorType)
        # copy date part
        state[:self.stateDevicesStartIdx] = dateTensor
        # calc expected state per device
        for i in range(self.numOfDevices):
            device = self.policyJSON[str(i)]
            deviceState = 0
            for timeDict in device:
                if dateObj.weekday() in timeDict["days"]:
                    for t in timeDict["times"]:
                        if (dateObj.time() >= datetime.strptime(t[0], self.policyJSON["Time format"]).time()) \
                                and (dateObj.time() <= datetime.strptime(t[1], self.policyJSON["Time format"]).time()):
                            deviceState = 1
                            break

            state[self.stateDevicesStartIdx + i] = deviceState

        return state

    def calculateReward(self, nextState, nextStateDateObj):
        assert (nextState.type() == self.tensorType)
        assert (len(nextState[self.stateDevicesStartIdx:]) == self.numOfDevices)
        # build expected next state
        expectedNextState = self.buildExpectedState(nextState[:self.stateDevicesStartIdx], nextStateDateObj)
        # calc devices state difference
        stateDiff = (nextState[self.stateDevicesStartIdx:] - expectedNextState[self.stateDevicesStartIdx:]).abs()
        # count mistakes & correct
        # TODO: count in Integer, rather than Float ???
        nMistakes = sum(stateDiff)
        nCorrect = self.numOfDevices - nMistakes

        # total reward is scaled in range [-1,1]
        reward = (nCorrect - nMistakes) / float(self.numOfDevices)
        return reward

    @staticmethod
    def loadPolicyFromJSON(fname):
        with open(fname, 'r') as f:
            policy = json.load(f)

        Policy.validatePolicy(policy)
        return policy

    @staticmethod
    def validatePolicy(policy):
        nDays = len(policy["days"])
        for i in range(len(policy["Devices"])):
            device = policy[str(i)]
            daysArray = [[] for j in range(nDays)]

            for timeDict in device:
                # replace predefined array in JSON with actual array for future simplicity
                if type(timeDict["days"]) is not list:
                    timeDict["days"] = policy[timeDict["days"]]

                # sort timeDict by startTime
                timeDict["times"] = sorted(timeDict["times"],
                                           key=lambda t: datetime.strptime(t[0], policy["Time format"]))

                for day in timeDict["days"]:
                    daysArray[day].extend(timeDict["times"])

            # sort time ranges for easier compare
            for j in range(len(daysArray)):
                daysArray[j] = sorted(daysArray[j], key=lambda t: datetime.strptime(t[0], policy["Time format"]))

            for array in daysArray:
                for j in range(len(array) - 1):
                    tCur = array[j]
                    tNext = array[j + 1]
                    if tCur[1] > tNext[0]:  # endTime is bigger than next range startTime
                        raise ValueError(
                            'Validation failed for device [{}], ID:[{}], time ranges: [{}] - [{}]'.format(
                                policy["Devices"][i], i, tCur, tNext))
