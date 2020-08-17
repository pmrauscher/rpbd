import numpy as np

########################### UTILITIES #######################################

def flattenTensor(tensor):
    if tensor.shape != (3,3):
        return np.zeros(6)
    else:
        result = np.zeros(6)
        result[0] = tensor[0][0] # xx
        result[1] = tensor[1][1] # yy
        result[2] = tensor[2][2] # zz
        result[3] = tensor[0][1] # xy
        result[4] = tensor[0][2] # xz
        result[5] = tensor[1][2] # yz
        return result
    
def unflattenTensor(tensor):
    if tensor.shape != (6,):
        return np.zeros((3,3))
    else:
        result = np.zeros((3,3))
        result[0,0] = tensor[0] # xx
        result[1,1] = tensor[1] # yy
        result[2,2] = tensor[2] # zz
        result[0,1] = result[1,0] = tensor[3] # xy
        result[0,2] = result[2,0] = tensor[4] # xz
        result[1,2] = result[2,1] = tensor[5] # yz
        return result


####################### SIMULATION CLASS ########################################

class Simulation:
    
    ########################## INITIALIZATION AND GETTER/SETTER FUNCTIONS #######################
    
    def __init__(self,numRings=1,beadsPerRing=8,hiType=0,calcEvery=100,writeEvery=1000,hStar=0.5,dt=0.01,output="out.xyz"):
        
        # clean input
        if numRings > 2:
            print("This code only supports [2]catenanes, larger constructs are not allowed. Number of rings reduced to two.")
            numRings = 2
        if numRings < 1:
            print("Not enough rings! Number of rings set to one.")
            numRings = 1
        if beadsPerRing < 3:
            print("Not enough beads per ring, at least three are needed. Setting beads per ring to eight (default).")
            beadsPerRing = 8
        
        # size of the system
        self.numRings = numRings
        self.beadsPerRing = beadsPerRing
        self.numBeads = numRings*beadsPerRing
        
        # run control
        self.dt = dt
        self.hiType = hiType # type of HI, 0 = freely-draining, 1 = pre-averaged, 2 = instantaneously averaged, 3 = full HI, 4 = ordinary pre-averaging
        self.calcEvery = calcEvery # how often to calculate HI tensor (if applicable)
        self.dev = np.sqrt(24.0/dt)
        self.output = output
        self.outfile = None
        self.writeEvery = writeEvery
        
        # hStar sets the strenght of HI, we use this to determine the 
        # spring constant for the bonds relative to the bead radius
        # see ref.: Zylka and Oettinger J.Chem.Phys. 90, 474 (1989)
        self.bondConst = hStar*hStar*np.pi
        
        # if the number of rings is greater than one, use a simple double-Rouse model
        # see ref.: Rauscher et al., J.Chem.Phys. 152, 214901 (2020)
        self.mechBondConst = 0.0
        if self.numRings > 1:
            self.mechBondConst = (12.0*self.bondConst)/float(self.beadsPerRing) # this will yield a bond length equal to the radius of gyration of the rings 
            
        # initializing vectors/matrices
        self.pos = np.zeros((self.numBeads,3))
        self.force = np.zeros((self.numBeads,3))
        self.noise = np.zeros((self.numBeads,3))
        self.mobility = np.eye(3*self.numBeads)
        self.brownian = np.eye(3*self.numBeads)
        
        # data structure for fourier transformed vectors (not necessary for all HI types)
        self.modePos = np.zeros((self.numBeads,3),dtype=np.cdouble)
        self.modeForce = np.zeros((self.numBeads,3),dtype=np.cdouble)
        self.modeNoise = np.zeros((self.numBeads,3),dtype=np.cdouble)
        
        
    
    # set/get positions, mobilities, and HI type
    def setPositions(self,pos):
        if pos.shape == self.pos.shape:
            self.pos = pos
            
    def getPositions(self):
        return self.pos
            
    def setMobility(self,mob):
        if mob.shape == self.mobility.shape:
            self.mobility = mob

    def getMobility(self):
        return self.mobility
        
    def setHI(self,hiType):
        self.hiType = hiType

    def getHI(self):
        return self.hiType
    
    def setCalc(self,calcEvery):
        self.calcEvery = calcEvery
        
    def setWrite(self,writeEvery):
        self.writeEvery = writeEvery

    def setOutput(self,output):
        self.output = output

    ############################ OUTPUT AND DATA RECORDING ############################
            
    def prepOutput(self):  
        self.outfile = open(self.output,"w")
        
    def closeOutput(self):  
        self.outfile.close()
    
    def writeCoords(self):
        self.outfile.write("{0}\nComment\n".format(self.numBeads))
        for i in range(self.numBeads):
            self.outfile.write("{0:.5f}\t{1:.5f}\t{2:.5f}\n".format(self.pos[i][0],self.pos[i][1],self.pos[i][2]))
        
    
    ################################# FORCE AND MOBILITY CALCULATIONS ###################    
        
    # calculate forces
    def calcForces(self):
        
        # reset forces
        self.force = np.zeros((self.numBeads,3))
        
        # get bonded forces within the rings
        for i in range(self.numRings):
            for j in range(self.beadsPerRing): 
                idx1 = i*self.beadsPerRing + j
                idx2 = i*self.beadsPerRing + (j+1)%self.beadsPerRing
                forceVec = self.bondConst*(self.pos[idx2] - self.pos[idx1])
                self.force[idx1] += forceVec
                self.force[idx2] -= forceVec

        # get inter-ring forces (if applicable)
        if self.numRings == 2:  
            com1 = np.mean(self.pos[0:self.beadsPerRing],axis=0)
            com2 = np.mean(self.pos[self.beadsPerRing:self.numBeads],axis=0)
            forceVec = self.mechBondConst*(com2 - com1)/float(self.beadsPerRing)
            for i in range(self.beadsPerRing):
                self.force[i] = np.add(self.force[i],forceVec)
                self.force[i+self.beadsPerRing] = np.subtract(self.force[i+self.beadsPerRing],forceVec)
        
    
    # get random values of appropriate variance
    def calcNoise(self):
        self.noise = self.dev*np.random.uniform(-0.5,0.5,(self.numBeads,3))
    
    
    # calculate HI matrix and get its decomposition
    def calcMobility(self,decomp=True):
        
        self.mobility = np.eye(3*self.numBeads)
        
        for i in range(self.numBeads):
            for j in range(i+1,self.numBeads):
                
                dr = self.pos[i] - self.pos[j]
                distSqr = np.sum(np.square(dr))
                dist = np.sqrt(distSqr)

                factor1 = factor2 = 0.0
                if (dist < 2.0):
                    factor1 = 1.0 - (9.0*dist)/32.0
                    factor2 = 3.0/(32.0*dist)
                else:
                    factor1 = (3.0/(4.0*dist))*(1.0 + 2.0/(3.0*distSqr))
                    factor2 = (3.0/(4.0*dist*distSqr))*(1.0 - 2.0/distSqr)
                    
                self.mobility[i*3:(i+1)*3,j*3:(j+1)*3] = self.mobility[j*3:(j+1)*3,i*3:(i+1)*3] = factor1*np.eye(3) + factor2*np.outer(dr,dr)
        
        if decomp:
            self.brownian = np.linalg.cholesky(self.mobility)
            
    
    # average the mobility tensor to make its blocks circulant then 
    # get its eigenvalues/tensors by fft and calculate square root matrices
    def processMobility(self):
        
        # intra-ring eigentensors and square roots
        eigIntra = np.zeros((3,3*self.numBeads))
        brownianIntra = np.zeros((3,3*self.numBeads))
        
        # inter-ring eigentensors and square roots
        eigInter = np.zeros((3*self.numRings,3*self.numRings))
        brownianInter = np.zeros((3,3*self.numBeads))
        
        # start with the intra-ring parts of the mobiltiy tensor
        mobilityIntra = np.zeros((self.numBeads,6))
        
        # for each ring
        for i in range(self.numRings):
            # for each possible separation (in terms of beads)
            for j in range(1,int(self.beadsPerRing/2)+1):
                # for each starting bead
                for k in range(self.beadsPerRing):
                    # add mobility to running total
                    idx1 = i*self.beadsPerRing + k
                    idx2 = i*self.beadsPerRing + (k+j)%self.beadsPerRing
                    interaction = flattenTensor(self.mobility[idx1*3:(idx1+1)*3,idx2*3:(idx2+1)*3])
                    mobilityIntra[i*self.beadsPerRing+j] += interaction
                    if j != int(int(self.beadsPerRing/2)):
                        mobilityIntra[i*self.beadsPerRing+self.beadsPerRing-j] += interaction
            
            # set diagonal totals
            mobilityIntra[i*self.beadsPerRing][0] = mobilityIntra[i*self.beadsPerRing][1] = mobilityIntra[i*self.beadsPerRing][2] = float(self.beadsPerRing)
        
        # average it out
        mobilityIntra = np.divide(mobilityIntra,float(self.beadsPerRing))
        
        # now Fourier transform the columns to get eigentensors
        # for each ring
        for i in range(self.numRings):
            # for each tensor element, fourier transform the column
            for j in range(6):
                mobilityIntra[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j] = np.real(np.fft.fft(mobilityIntra[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j]))
            
            # further processing
            for j in range(1,self.beadsPerRing):
                # unflatten the eigentensors
                eigIntra[0:3,(i*self.beadsPerRing+j)*3:(i*self.beadsPerRing+j+1)*3] = unflattenTensor(mobilityIntra[i*self.beadsPerRing+j])
                # get the square root via cholesky decomp
                brownianIntra[0:3,(i*self.beadsPerRing+j)*3:(i*self.beadsPerRing+j+1)*3] = np.linalg.cholesky(eigIntra[0:3,(i*self.beadsPerRing+j)*3:(i*self.beadsPerRing+j+1)*3])
                
        # now deal with inter-ring HI averaging
        # for each pair of rings
        for i in range(self.numRings):
            for j in range(i+1,self.numRings):
                interaction = np.zeros((3,3))
                # for each pair of beads
                for k in range(self.beadsPerRing):
                    for l in range(self.beadsPerRing):
                        idx1 = i*self.beadsPerRing + k
                        idx2 = j*self.beadsPerRing + l
                        interaction += self.mobility[idx1*3:(idx1+1)*3,idx2*3:(idx2+1)*3]
                
                # average it out - should divide by m^2 but the fft adds a factor of m so we include that here too
                interaction /= float(self.beadsPerRing)
                eigInter[i*3:(i+1)*3,j*3:(j+1)*3] = eigInter[j*3:(j+1)*3,i*3:(i+1)*3] = interaction
                
            # set diagonal block eigentensors
            eigInter[i*3:(i+1)*3,i*3:(i+1)*3] = unflattenTensor(mobilityIntra[i*self.beadsPerRing])
        
        # get decomposition of inter-ring portions
        brownianInter = np.linalg.cholesky(eigInter)
        
        # return the results
        return eigIntra,brownianIntra,eigInter,brownianInter
    
    
    
    
    
    ####################### COORDINATE TRANSFORMATIONS #############################
    
    # move coordinates, forces, and noise to Fourier space by FFT
    def toFourier(self):
        
        # for each ring
        for i in range(self.numRings):
            # for each spatial dimension
            for j in range(3):
                self.modePos[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j] = np.fft.fft(self.pos[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j])
                self.modeForce[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j] = np.fft.fft(self.force[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j])
                self.modeNoise[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j] = np.fft.fft(self.noise[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j])
    
    # move coordinates to real space by inverse FFT
    # only need to do the positions as forces and noise will be wiped and re-calculated each step
    def toReal(self):
        
        # for each ring
        for i in range(self.numRings):
            # for each spatial dimension
            for j in range(3):
                self.pos[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j] = np.real(np.fft.ifft(self.modePos[i*self.beadsPerRing:(i+1)*self.beadsPerRing,j]))
                
    
    
    
    
    ######################## INTEGRATORS AND ALGORITHMS ###################################
    
    # integrator for freely-draining simulations
    def integrateFD(self,numSteps):
        
        # place to store average mobility tensor (for use in pre-avg sims later on)
        avgMobility = np.zeros(self.mobility.shape)
        count = 0
        
        # do force calculation
        self.calcForces()
        self.calcNoise()
        
        # Euler integration
        for step in range(numSteps):
            self.pos += self.dt*np.add(self.force, self.noise)
            self.calcForces()
            self.calcNoise()
            
            # calculate the mobility every few steps for pre-averaging
            if (step+1)%self.calcEvery == 0:
                self.calcMobility(False)
                avgMobility += self.getMobility()
                count += 1
                
            # write coordinates
            if (step+1)%self.writeEvery == 0:
                self.writeCoords()
        
        if count > 0:
            avgMobility = np.divide(avgMobility,float(count))
            return avgMobility
        return 0
    
    # integrator for pre-averaged simulations
    def integratePre(self,numSteps):

        # check that the mobility tensor has been set
        if np.allclose(self.mobility,np.eye(3*self.numBeads)):
            print("The mobility tensor has not been set. Simulation aborted.")
            return 1
        
        # get the intra and inter-ring HI eigenvalues
        eigIntra, brownianIntra, eigInter, brownianInter = self.processMobility()
        
        # do force calculation
        self.calcForces()
        self.calcNoise()
        
        # now run the simulation
        for step in range(numSteps):
            
            # move to Fourier space
            self.toFourier()
            
            # update the moves q > 0 (internal modes)
            # for each ring
            for i in range(self.numRings):
                for j in range(1,self.beadsPerRing):
                    idx = i*self.beadsPerRing + j
                    # get displacement due to forces
                    dispForce = np.matmul(eigIntra[0:3,idx*3:(idx+1)*3],np.transpose(self.modeForce[idx]))
                    # get displacement due to noise
                    dispNoise = np.matmul(brownianIntra[0:3,idx*3:(idx+1)*3],np.transpose(self.modeNoise[idx]))
                    # update the mode
                    self.modePos[idx] += self.dt*(dispForce + dispNoise)
            
            # update the COM modes (q = 0)
            # move to more convenient data shape
            comPos = np.array([self.modePos[i*self.beadsPerRing] for i in range(self.numRings)]).reshape((self.numRings*3,1))
            comForce = np.array([self.modeForce[i*self.beadsPerRing] for i in range(self.numRings)]).reshape((self.numRings*3,1))
            comNoise = np.array([self.modeNoise[i*self.beadsPerRing] for i in range(self.numRings)]).reshape((self.numRings*3,1))
            
            # update COM positions and reshape
            comPos += self.dt*(np.matmul(eigInter,comForce) + np.matmul(brownianInter,comNoise))
            comPos = comPos.reshape((self.numRings,3))
            # put COM positions back in the array
            for i in range(self.numRings):
                self.modePos[i*self.beadsPerRing] = comPos[i]
                
            # move to real space
            self.toReal()
            
            # calculate noise and forces
            self.calcForces()
            self.calcNoise()   
            
            # write coordinates
            if (step+1)%self.writeEvery == 0:
                self.writeCoords()
        
        return 0
    
    
    # integrator for instantaneously-averaged simulations
    def integrateInst(self,numSteps):

        # calculcate mobility and get the intra and inter-ring HI eigenvalues
        self.calcMobility(False)
        eigIntra, brownianIntra, eigInter, brownianInter = self.processMobility()
        
        # do force calculation
        self.calcForces()
        self.calcNoise()
        
        # now run the simulation
        for step in range(numSteps):
            
            # move to Fourier space
            self.toFourier()
            
            # update the moves q > 0 (internal modes)
            # for each ring
            for i in range(self.numRings):
                for j in range(1,self.beadsPerRing):
                    idx = i*self.beadsPerRing + j
                    # get displacement due to forces
                    dispForce = np.matmul(eigIntra[0:3,idx*3:(idx+1)*3],np.transpose(self.modeForce[idx]))
                    # get displacement due to noise
                    dispNoise = np.matmul(brownianIntra[0:3,idx*3:(idx+1)*3],np.transpose(self.modeNoise[idx]))
                    # update the mode
                    self.modePos[idx] += self.dt*(dispForce + dispNoise)
            
            # update the COM modes (q = 0)
            # move to more convenient data shape
            comPos = np.array([self.modePos[i*self.beadsPerRing] for i in range(self.numRings)]).reshape((self.numRings*3,1))
            comForce = np.array([self.modeForce[i*self.beadsPerRing] for i in range(self.numRings)]).reshape((self.numRings*3,1))
            comNoise = np.array([self.modeNoise[i*self.beadsPerRing] for i in range(self.numRings)]).reshape((self.numRings*3,1))
            
            # update COM positions and reshape
            comPos += self.dt*(np.matmul(eigInter,comForce) + np.matmul(brownianInter,comNoise))
            comPos = comPos.reshape((self.numRings,3))
            # put COM positions back in the array
            for i in range(self.numRings):
                self.modePos[i*self.beadsPerRing] = comPos[i]
                
            # move to real space
            self.toReal()
            
            # calculate noise and forces
            self.calcForces()
            self.calcNoise()
            
            # calculate the mobility every few steps
            if (step+1)%self.calcEvery == 0:
                self.calcMobility(False)
                eigIntra, brownianIntra, eigInter, brownianInter = self.processMobility()
                
            # write coordinates
            if (step+1)%self.writeEvery == 0:
                self.writeCoords()
        
        return 0
    
        
    
    # integrator for full HI simulations
    def integrateFull(self,numSteps):
        
        # calculate mobility
        self.calcMobility(True)
        
        # do force calculation
        self.calcForces()
        self.calcNoise()
        
        # Euler integration
        for step in range(numSteps):
            displacement = self.dt*(np.matmul(self.mobility,self.force.reshape((3*self.numBeads,1))) + np.matmul(self.brownian,self.noise.reshape((3*self.numBeads,1))))
            self.pos += displacement.reshape((self.numBeads,3))
            self.calcForces()
            self.calcNoise()
                                 
            # calculate the mobility every few steps for pre-averaging
            if (step+1)%self.calcEvery == 0:
                self.calcMobility(True)
                
            # write coordinates
            if (step+1)%self.writeEvery == 0:
                self.writeCoords()
        
        return 0
    
    # integrator for pre-averaged sims using the naive algorithm
    def integratePreOrdinary(self,numSteps):
        
        # check that the mobility tensor has been set
        if np.allclose(self.mobility,np.eye(3*self.numBeads)):
            print("The mobility tensor has not been set. Simulation aborted.")
            return 1
        
        # get brownian matrix
        self.brownian = np.linalg.cholesky(self.getMobility())
        
        # do force calculation
        self.calcForces()
        self.calcNoise()
        
        # Euler integration
        for step in range(numSteps):
            displacement = self.dt*(np.matmul(self.mobility,self.force.reshape((3*self.numBeads,1))) + np.matmul(self.brownian,self.noise.reshape((3*self.numBeads,1))))
            self.pos += displacement.reshape((self.numBeads,3))
            self.calcForces()
            self.calcNoise()
                
            # write coordinates
            if (step+1)%self.writeEvery == 0:
                self.writeCoords()
        
        return 0
        
    
    # run the simulation
    def run(self,numSteps):
        
        # check that positions have been set
        if np.allclose(self.pos,np.zeros((self.numBeads,3))):
            print("Bead positions have not been set. Simulation aborted.")
            return 1
        
        # open output file
        self.prepOutput()
        
        # run the appropriate algorithm
        if self.hiType == 0:
            result =  self.integrateFD(numSteps)
        elif self.hiType == 1:
            result =  self.integratePre(numSteps)
        elif self.hiType == 2:
            result =  self.integrateInst(numSteps)
        elif self.hiType == 3:
            result =  self.integrateFull(numSteps)
        elif self.hiType == 4:
            result =  self.integratePreOrdinary(numSteps)
        else:
            print("Could not determine what kind of HI to implement. Simulation aborted.")
            result = 1
            
        self.closeOutput()
        return result
    
    
