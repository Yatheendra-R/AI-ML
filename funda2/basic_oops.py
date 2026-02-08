"""
| Concept            | Java / C++                 | Python                        |
| ------------------ | -------------------------- | ----------------------------- |
| Class              | `class A {}`               | `class A:`                    |
| Object             | `A a = new A()`            | `a = A()`                     |
| Constructor        | `A()`                      | `__init__()`                  |
| `this`             | `this`                     | `self`                        |
| Access modifiers   | `public/private/protected` | Naming convention (`_`, `__`) |
| Method overloading | Compile-time               | ❌ (default args instead)      |
| Method overriding  | ✔                          | ✔                             |
| Interfaces         | `interface`                | `abc` module                  |
| Destructor         | `~A()`                     | `__del__()` (rarely used)     |

"""
#self = current object (like this)
class Student:
    def __init__(self, age):
        self.age = age
        self.a = 10      # public
        self._b = 20     # protected (convention)
        self.__c = 30    # private (name mangling)
    def printing(self,age1):
        print(self.age)
        print(age1)
    

s = Student(10)
s.printing(100)

#single inheritance 
class Parent:
    def __init__(self):
        print("Parent construct")
    def show(self):
        print("Parent show")
class Child(Parent):             #Inheritance
    def __init__(self):
        super().__init__()  #No. Python does NOT implicitly call the parent constructor when the child defines its own __init__.
        #calling parent construct
        print("child construct")
    def show(self):                         #Method overriding
        print("child show")
        super().show()    #calling parent method, which has been overriden
        

CH=Child()
CH.show()
print("mro: ",Child.mro()) #mro() is a class method, not an instance method.

#Multiple Inheritance
class A:
    def show(self):
        print("A")

class B:
    def show(self):
        print("B")

class C(A, B):
    def show(self):
        
        print("C")
        super().show()



c = C()
c.show()
print("mro: ",C.mro()) # mro class C(A, B) → left-to-right        , C → A → B → object
"""
Critical misunderstanding to avoid -> super() does NOT mean “call parent class”

It means: Call the next class in MRO after the current class
"""

print("\n")
#hybrid single  + multiple   , dimond problem
"""
    A
   / \
  B   C
   \ /
    D

"""
class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")
        super().show()

class C(A):
    def show(self):
        print("C")
        super().show()

class D(B, C):
    def show(self):
        print("D")
        super().show()

d = D()
d.show()
print("mro: ",D.mro())   #D->B->C->A

"""
Multilevel inheritance

Inheritance where a class derives from another derived class.

Multiple inheritance

A class inheriting from more than one base class.

Ambiguity

When multiple parent classes contain the same method name.

Python’s solution

Python resolves ambiguity using Method Resolution Order (MRO).
"""
print()

#Multilevel Inheritance (NO ambiguity)
#Structure:   GrandParent → Parent → Child
class GrandParent:
    def show(self):
        print("GrandParent")

class Parent(GrandParent):
    def show(self):
        print("Parent")

class Child(Parent):
    def show(self):
        print("Child")

c = Child()
c.show()
    


