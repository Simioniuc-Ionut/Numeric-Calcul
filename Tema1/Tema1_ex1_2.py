#Exercitiul 1
m = 0.0
u = 10.0 **(-m)
print(u)
while 1 + u != 1:
  m += 1
  u = 10 **(-m)
  
m -= 1.0
u = 10.0 **(-m)
print(f"value of u = {u} and value of m =  {m}")
print("-"*50)
#Exercitiul 2
x= 1.0
y = u/10
z = u/10

print((x+y) + z)
print(x + (y + z))

print("Operation +c is associative :", (x+y) + z == x+ (y + z))
print(f"Value of y = {y}, Value of z = {z}")

x = u / 10
y = u / 10
z = u / 10
m = 0
u = 10 ** (-m)
while (x*y) * z == x * (y*z):
  m += 1
  u = 10 ** (-m)
  y = u / 10
  z = u / 10

print(f"value of u = {u} and value of m =  {m}")
print(f"Multiplication Operation is associative :{(x * y) * z == x * (y * z)}" )