from Simulation import Simulation

def main():
    simulation = Simulation()
    #
    simulation.initClassifier()
    simulation.testClassifier()

    # simulation.signalEncoder.plotEncodedSpikesSignals()
    # simulation.run()
if __name__ == "__main__":
    main()
