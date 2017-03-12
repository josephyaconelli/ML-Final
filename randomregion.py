import random
import sys

def main(argv):
  r1 = ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont']
  r2 = ['New Jersey', 'New York']
  r3 = ['Delaware', 'Maryland', 'Pennslyvania', 'Virginia', 'West Virginia']
  r4 = ['Alabama', 'Florida', 'Georgia', 'Kentucky', 'Mississippi', 'North Carolina', 'South Carolina', 'Tennessee']
  r5 = ['Illinois', 'Indiana', 'Michigan', 'Minnesota', 'Ohio', 'Wisconsin']
  r6 = ['Arkansas', 'Louisiana', 'New Mexico', 'Oklahoma', 'Texas']
  r7 = ['Iowa', 'Kansas', 'Missouri', 'Nebraska']
  r8 = ['Colorado', 'Montana', 'North Dakota', 'South Dakota', 'Utah', 'Wyoming']
  r9 = ['Arizona', 'California', 'Hawaii', 'Nevada']
  r10 = ['Alaska', 'Idaho', 'Oregon', 'Washington']
  #print (len(r1) + len(r2) + len(r3) + len(r4) + len(r5) + len(r6) + len(r7) + len(r8) + len(r9) + len(r10))
  print(random.choice(r1))
  print(random.choice(r2))
  print(random.choice(r3))
  print(random.choice(r4))
  print(random.choice(r5))
  print(random.choice(r6))
  print(random.choice(r7))
  print(random.choice(r8))
  print(random.choice(r9))
  print(random.choice(r10))


if __name__ == "__main__":
  main(sys.argv[1:])
