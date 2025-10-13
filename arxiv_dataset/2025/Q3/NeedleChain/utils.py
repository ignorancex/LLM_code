import os
import json
import operator
import random
import copy
from functools import reduce


NAMES = ['Rhett', 'Cillian', 'Ana', 'Rosa', 'Malik', 'Saul', 'Kashton', 'Kataleya', 'Ellianna', 'Amirah', 'Leonardo', 'Kaizen', 'Karsyn', 'Jireh', 'Marcos', 'Samson', 'Calvin', 'Cleo', 'Weston', 'Samantha', 'Waylen', 'Liv', 'Rayan', 'Colin', 'Gwendolyn', 'Leon', 'Colter', 'Callen', 'Alaia', 'Jagger', 'Malka', 'Karter', 'Alonso', 'Forrest', 'Alex', 'Jesiah', 'Stephen', 'Christina', 'Vance', 'Cecelia', 'Katie', 'Harry', 'Peyton', 'Michael', 'Athena', 'Ashlyn', 'Mack', 'Mary', 'Galilea', 'Honey', 'Karla', 'Victor', 'Fernando', 'Halle', 'Quinton', 'Landen', 'Hadlee', 'Jaxson', 'Adele', 'Treasure', 'Madilyn', 'Miranda', 'Fletcher', 'Gemma', 'Madelynn', 'Blakely', 'Titus', 'Evie', 'Barrett', 'Amanda', 'Haven', 'Alonzo', 'Eliam', 'Benjamin', 'Mario', 'Azariah', 'Angela', 'Zavier', 'Wren', 'Kennedy', 'Astrid', 'Julietta', 'Giuliana', 'Reign', 'Anya', 'Arabella', 'Hector', 'Porter', 'Sasha', 'Lawson', 'Jedidiah', 'Journee', 'Kylie', 'Emma', 'Kallie', 'Sydney', 'Rosalina', 'Octavia', 'Zoya', 'Averie', 'Lucille', 'Amina', 'Hailey', 'Emilio', 'Ivey', 'Novah', 'Blake', 'Gerardo', 'Elsa', 'Louie', 'Rachel', 'Julio', 'Murphy', 'Jenesis', 'Laila', 'Maxine', 'Zaniyah', 'Aubrey', 'Misael', 'Matheo', 'Chloe', 'Omar', 'Wylder', 'Lucy', 'Bodhi', 'True', 'Noemi', 'Ashley', 'Jianna', 'Leonidas', 'Danielle', 'Adrian', 'Cohen', 'Adhara', 'Owen', 'Rodrigo', 'Saanvi', 'Francis', 'Aidan', 'Kaiden', 'Uriel', 'Elsie', 'Victoria', 'Adelaide', 'Deacon', 'Nylah', 'Genesis', 'Siya', 'Rowan', 'Clementine', 'Luella', 'Dean', 'Shane', 'Zymir', 'Makai', 'Jakari', 'Dustin', 'Ryland', 'Valentin', 'Bode', 'Mckenna', 'Lilian', 'Nina', 'Vada', 'Janiyah', 'Teddy', 'Emerie', 'Addison', 'Saint', 'Calum', 'Madalyn', 'Ezrah', 'Emmeline', 'Mckenzie', 'Ridge', 'Lakelyn', 'Gage', 'Ali', 'Nadia', 'Rafael', 'Kori', 'Makayla', 'Zelda', 'Zev', 'Aubree', 'Matias', 'Noa', 'Helen', 'Jasper', 'Ronald', 'Londyn', 'Boden', 'Prince', 'Adan', 'Guillermo', 'Nico', 'Hadassah', 'Veronica', 'Banks', 'Clyde', 'Dawson', 'Benedict', 'Kylan', 'Ariya', 'Laurel', 'Azalea', 'Leona', 'Silas', 'Rylan', 'Callie', 'Dylan', 'Maelynn', 'Collins', 'Kace', 'Presley', 'Wayne', 'Alice', 'Zaylee', 'Mila', 'Carlo', 'Reya', 'Nalani', 'Kahlani', 'Lenora', 'Brock', 'Skyler', 'Payton', 'Lavender', 'Kyren', 'Alexander', 'Jonas', 'Boston', 'Lucas', 'Maximo', 'Zainab', 'Franklin', 'Jimmy', 'Cassandra', 'Enoch', 'Selah', 'Gian', 'Avani', 'Natalie', 'Rosalee', 'Wrenley', 'Ivanna', 'Loretta', 'Lochlan', 'Dakota', 'Keily', 'Brielle', 'Jamari', 'Keanu', 'Ramon', 'Maddie', 'Briggs', 'Jazmine', 'Vienna', 'Asaiah', 'Mina', 'Jay', 'Lilia', 'Kareem', 'Cannon', 'Moses', 'Avi', 'Adaline', 'Priscilla', 'Anaiah', 'Lina', 'Joshua', 'Jade', 'Arjun', 'Sunny', 'Monica', 'Amaia', 'Morgan', 'Nevaeh', 'Emi', 'Abdiel', 'Emmanuel', 'Iyla', 'Audrey', 'Maximilian', 'Knox', 'Kingston', 'Freyja', 'Fatima', 'Nicholas', 'Bridger', 'Trey', 'Delilah', 'Kayden', 'Jorge', 'Aliyah', 'Justin', 'Ainhoa', 'Collin', 'Gwen', 'Beckett', 'Jessica', 'Lachlan', 'Keziah', 'Adelynn', 'Grace', 'Valerie', 'Briana', 'Carson', 'Grant', 'Diana', 'Francesca', 'Edgar', 'Layton', 'Desmond', 'Greyson', 'Matthias', 'Anthony', 'Marleigh', 'Santana', 'Yosef', 'Raya', 'Atreus', 'Sullivan', 'Nazir', 'Hezekiah', 'Lucian', 'Rayne', 'Jared', 'Dimitri', 'Georgina', 'Anderson', 'Crew', 'Darian', 'Riley', 'Oaklynn', 'Melanie', 'Guinevere', 'Kendrick', 'Aiden', 'Samara', 'Rebekah', 'Nancy', 'Madilynn', 'Luisa', 'Ella', 'Rhea', 'Kensley', 'Archer', 'Adriana', 'Callan', 'Noor', 'Lydia', 'Katherine', 'Paxton', 'Zoe', 'Bradley', 'Savannah', 'Alyssa', 'Josiah', 'Nala', 'Maisie', 'Esme', 'Billie', 'Conrad', 'Neil', 'Berkley', 'Carolina', 'Hendrix', 'Memphis', 'Gia', 'Yehuda', 'Maeve', 'Capri', 'Bennett', 'Nataly', 'Luciana', 'Sincere', 'Blair', 'Charley', 'Joziah', 'Darwin', 'Elyse', 'Bryce', 'Allyson', 'Alani', 'Elena', 'Noe', 'Davis', 'Nixon', 'Ameer', 'Tanner', 'Elouise', 'Rayna', 'Chozen', 'Mavis', 'Valentina', 'Yara', 'Miriam', 'Sarah', 'Kellan', 'Cattleya', 'Eleanora', 'Luciano', 'Saige', 'Kash', 'Watson', 'Kai', 'Madison', 'Osiris', 'Bridget', 'Mccoy', 'Ariah', 'Kian', 'Lilah', 'Shimon', 'Ayaan', 'Gianna', 'Naya', 'Ember', 'Clare', 'Bonnie', 'Damien', 'Shawn', 'Kyra', 'Karim', 'Westin', 'Krue', 'Meir', 'Ensley', 'Frances', 'Valery', 'Kyla', 'Bryer', 'Gael', 'Stella', 'Sebastian', 'Phoebe', 'Raelynn', 'Rowyn', 'Poppy', 'Alfredo', 'Michaela', 'Izael', 'Troy', 'Dane', 'Dorian', 'Aldo', 'Kolson', 'Legend', 'Alejandro', 'Indy', 'Malani', 'Dillon', 'Mohammad', 'Andrew', 'Kassidy', 'Ethan', 'Clover', 'Bodie', 'Edith', 'Diego', 'Shelby', 'Ila', 'Estella', 'Salem', 'Landyn', 'Margaret', 'Enzo', 'Destiny', 'Charles', 'Kenji', 'Curtis', 'Karson', 'Hope', 'Fisher', 'Alejandra', 'Tiffany', 'Brooklynn', 'Lottie', 'Gabriela', 'Sutton', 'Alanna', 'Julieta', 'Alayah', 'Muhammad', 'Lorenzo', 'Beckham', 'Koda', 'Tony', 'Rylee', 'Marina', 'Jane', 'Shepherd', 'Bo', 'Giana', 'David', 'Logan', 'Aileen', 'Ariana', 'Gabrielle', 'Solana', 'Violet', 'Leila', 'Antonio', 'Joelle', 'Benny', 'Isael', 'Amara', 'Zane', 'Kabir', 'Maurice', 'Aurora', 'Adonis', 'Ivy', 'Eleanor', 'Arlet', 'Martha', 'Lilianna', 'Joey', 'Camila', 'Ava', 'Scott', 'Zyair', 'Zahra', 'Angelo', 'Yamileth', 'Marilyn', 'Carmelo', 'Annie', 'Alora', 'Catalina', 'Creed', 'Lea', 'Kaylee', 'Kameron', 'Myra', 'Scarlett', 'Andie', 'Baker', 'Leighton', 'Halo', 'Mauricio', 'Samuel', 'Zion', 'Atlas', 'Alexa', 'Jaxxon', 'Azaria', 'Heaven', 'Leonard', 'Khaza', 'Freya', 'Elowen', 'Delaney', 'Kasen', 'Camilla', 'Hamza', 'Tobias', 'Asher', 'Ronnie', 'Chris', 'Autumn', 'Zechariah', 'Emmett', 'Everlee', 'Meilani', 'Ainsley', 'Lisa', 'Kenzo', 'Seth', 'Zen', 'Imani', 'Augustine', 'Niko', 'Jax', 'Leonel', 'Frank', 'Penelope', 'Macie', 'Axel', 'Kolton', 'Rey', 'Katalina', 'Aniyah', 'Gracelynn', 'Kora', 'Daniela', 'Malaya', 'Leland', 'Cora', 'Elianna', 'Koa', 'Jennifer', 'Megan', 'Emiliana', 'Lennox', 'Ryker', 'Rio', 'Kenzie', 'Julian', 'Azael', 'Sky', 'Anika', 'Emory', 'Bethany', 'Amos', 'Thalia', 'Rene', 'Cynthia', 'Catherine', 'Holly', 'Kaleb', 'Naomi', 'Tessa', 'Aaron', 'Israel', 'Azrael', 'Shlomo', 'Leilany', 'Amelie', 'Aden', 'Jeremy', 'Alicia', 'Jazmin', 'Zayn', 'Angie', 'Ellie', 'Giovanni', 'Ivory', 'Carter', 'Ryleigh', 'Ayan', 'Malachi', 'Dahlia', 'Messiah', 'Luka', 'Douglas', 'Eliza', 'Scarlet', 'Danna', 'Baylor', 'Faith', 'Jocelyn', 'Brycen', 'Henrik', 'Anastasia', 'Joy', 'Holden', 'Omari', 'Jaylen', 'Raegan', 'Paislee', 'Amy', 'Margo', 'Indigo', 'Kingsley', 'Maxwell', 'Molly', 'Anais', 'Jettson', 'Musa', 'Luca', 'Valentino', 'Aliza', 'Kylen', 'Dariel', 'Ariyah', 'Sloane', 'Jeremiah', 'Brianna', 'Ibrahim', 'Roman', 'Jesse', 'Winston', 'Kane', 'Arely', 'Anakin', 'Lucca', 'Marvin', 'Grayson', 'Yahya', 'Melany', 'Amiri', 'Mitchell', 'Thomas', 'Allan', 'Walker', 'Franco', 'Luz', 'Dexter', 'Mara', 'Ray', 'Elizabeth', 'Ruth', 'Oscar', 'Ricky', 'Alaiya', 'Krew', 'Lena', 'Julius', 'Cali', 'Dangelo', 'William', 'Davina', 'Paisley', 'Alan', 'Layla', 'Augustus', 'Vincent', 'Lucien', 'Nola', 'Brooks', 'Kenneth', 'Mathew', 'Ace', 'Julia', 'Alijah', 'Madden', 'Eren', 'Julianna', 'Deborah', 'Connor', 'Marley', 'Aleah', 'Dalton', 'Kalani', 'Charlee', 'Marcel', 'Louise', 'Izaiah', 'Joseph', 'Frederick', 'Greta', 'Blaire', 'Khalani', 'Colson', 'Princeton', 'Belen', 'Rocky', 'Andi', 'Eloise', 'Eliseo', 'Analia', 'Mariah', 'Aleena', 'Mylah', 'Julien', 'Kyle', 'Garrett', 'Persephone', 'Teagan', 'Janelle', 'Lainey', 'Dereck', 'Castiel', 'Reese', 'Laylani', 'Skyla', 'Cheyenne', 'Azai', 'Van', 'Brynlee', 'Celina', 'Aleia', 'Wesson', 'Jackson', 'Abigail', 'Rhys', 'Otis', 'Tru', 'Zachary', 'Emily', 'Joanna', 'Jerry', 'Andy', 'Emmitt', 'Navy', 'Derek', 'Hayden', 'Phoenix', 'Mason', 'Clay', 'Zayden', 'Edwin', 'Tristan', 'Mustafa', 'Graham', 'Raiden', 'Cayden', 'Henry', 'Atticus', 'Jaxon', 'Mikayla', 'Lilly', 'Elio', 'Isaac', 'Walter', 'Lee', 'Neriah', 'Armando', 'Felix', 'Anahi', 'Abraham', 'Phillip', 'Kobe', 'Tallulah', 'Eliana', 'Daisy', 'Rebecca', 'Noelle', 'Beatrice', 'Remington', 'Massimo', 'Joaquin', 'Elliott', 'Dorothy', 'Emmy', 'Jacqueline', 'Donald', 'Hanna', 'Soraya', 'Johan', 'Lila', 'Cristian', 'Whitley', 'Leandro', 'Andres', 'Xiomara', 'Charlotte', 'Noah', 'Max', 'Emberly', 'Emerson', 'Raina', 'Matilda', 'Maddison', 'Harmony', 'Adam', 'Meredith', 'Evelynn', 'Ben', 'Ulises', 'Gustavo', 'Wes', 'Elian', 'Sonny', 'Onyx', 'Dalia', 'Stormi', 'Jamison', 'Romy', 'Moises', 'Felipe', 'Mabel', 'Aurelia', 'Kyrie', 'Sage', 'Yaakov', 'Andrea', 'Demi', 'Liam', 'Ahmed', 'Indie', 'Judith', 'Ozzy', 'Dario', 'Zain', 'Austin', 'Tate', 'Gracie', 'Dax', 'Avyaan', 'Elina', 'Jenna', 'Major', 'Zyon', 'Khai', 'Orion', 'Yeshua', 'Kohen', 'Camden', 'Chelsea', 'Brantley', 'Myla', 'Brooklyn', 'Paige', 'Liberty', 'Mohammed', 'Milana', 'Scottie', 'Skylar', 'Adalynn', 'Kaiya', 'Dennis', 'Hayes', 'Orlando', 'Jordyn', 'Ernesto', 'Tyson', 'Cason', 'Jaime', 'Izabella', 'Amaya', 'Analeia', 'Angelina', 'Dayana', 'Yaretzi', 'Dash', 'Marcus', 'Itzel', 'Jaziel', 'Giselle', 'Jasiel', 'Cecilia', 'Keira', 'Paulina', 'Rayden', 'Jase', 'Hank', 'Kevin', 'Serenity', 'Caleb', 'Kiara', 'Sierra', 'Kaia', 'Wrenlee', 'Kylian', 'Kaisen', 'Waverly', 'Eileen', 'Giovanna', 'Eithan', 'Arya', 'Nathan', 'Amari', 'Donovan', 'Wyatt', 'Charleigh', 'Fallon', 'Alden', 'Myles', 'Edward', 'Allen', 'Maryam', 'Brixton', 'Wells', 'Veda', 'Alison', 'Waylon', 'Cairo', 'Kira', 'Remi', 'Genevieve', 'Mariam', 'Heath', 'Maddox', 'Soren', 'Emelia', 'Jacob', 'Braylen', 'Esteban', 'Gabriella', 'Agustin', 'Ander', 'Robin', 'Timothy', 'Zahir', 'Lewis', 'June', 'Eiden', 'Jakai', 'Elani', 'Junior', 'Roberto', 'Gloria', 'Aisha', 'Roger', 'Teo', 'Inaya', 'Marceline', 'Gatlin', 'Taytum', 'Killian', 'Aria', 'Aya', 'Kelsey', 'Winter', 'Carly', 'Amira', 'Ivan', 'Royalty', 'Sol', 'Isabella', 'Bailey', 'Journey', 'Ronin', 'Melvin', 'Thea', 'Lexi', 'Aila', 'Cassidy', 'Hunter', 'Paloma', 'Wallace', 'Emely', 'Kennedi', 'Elle', 'Elise', 'Cruz', 'Raphael', 'Alana', 'Tripp', 'Mikaela', 'Gianni', 'Scout', 'Adelyn', 'Jonathan', 'Harmoni', 'Erik', 'Elodie', 'Apollo', 'Kiaan', 'Laura', 'Haley', 'Promise', 'Dominic', 'Sam', 'Jayson', 'Jolie', 'Araceli', 'Aries', 'Christopher', 'Nora', 'Chase', 'Madeline', 'Heidi', 'Briar', 'Stephanie', 'Alena', 'Zander', 'Malaysia', 'Eric', 'Marisol', 'Emerald', 'Skye', 'Cielo', 'Lyra', 'Magnolia', 'Allison', 'Alexandria', 'Lyla', 'Arian', 'Liliana', 'Ezekiel', 'Iker', 'Olive', 'Eliel', 'Vihaan', 'Dominick', 'Kyson', 'April', 'Simon', 'Jayleen', 'Sara', 'Maisy', 'Norah', 'Trinity', 'Forest', 'Evren', 'Isabelle', 'Jaxton', 'Keith', 'Willow', 'Fernanda', 'Iliana', 'Lauren', 'Theo', 'Marianna', 'Jefferson', 'Darius', 'Alayna', 'Odin', 'Cataleya', 'Louisa', 'Miles', 'Tucker', 'Adriel', 'Declan', 'Gregory', 'Macy', 'Lincoln', 'Cesar', 'Charlie', 'Francisco', 'Huxley', 'Zayla', 'Gideon', 'Nikolai', 'Bryan', 'Ayah', 'Jolene', 'Maren', 'Kenai', 'Keegan', 'Zara', 'Brian', 'Hallie', 'Reid', 'Jiraiya', 'Stanley', 'Aaliyah', 'Isla', 'Oakley', 'Carmen', 'Aliya', 'Roy', 'Xyla', 'Carlos', 'Vivienne', 'Willa', 'Alianna', 'Jaiden', 'Azari', 'Raven', 'Claire', 'Abram', 'Zariyah', 'Xavier', 'Anne', 'Milan', 'Zamir', 'Kailany', 'Camille', 'Brooke', 'Truce', 'Amani', 'Rose', 'Maximiliano', 'Noel', 'Rivka', 'Kyaire', 'Linda', 'Melissa', 'Hazel', 'Iris', 'Alessio', 'Elliot', 'Maliyah', 'Sawyer', 'Malakai', 'Kehlani', 'Marjorie', 'Adalyn', 'Kamila', 'Yitzchok', 'Oakleigh', 'Avery', 'Zaylen', 'Jericho', 'Luna', 'Zoey', 'Robert', 'Lakelynn', 'Seraphina', 'Leia', 'Jeffrey', 'Arielle', 'Annalise', 'Kason', 'Jasmine', 'Samir', 'Jimena', 'Finnegan', 'Campbell', 'Tommy', 'Flora', 'Byron', 'Evan', 'Maci', 'Marcelo', 'Kaliyah', 'Magnus', 'Katelyn', 'Annika', 'Bellamy', 'Rosemary', 'Ruben', 'Rosalie', 'Brynleigh', 'Benicio', 'Sylas', 'Rome', 'George', 'Tadeo', 'Arlo', 'Harley', 'Elisa', 'Quincy', 'Ada', 'Kaiser', 'Jahmir', 'Daxton', 'Jordan', 'Erin', 'Sean', 'Lyric', 'Zyaire', 'Malia', 'Adelina', 'Emberlynn', 'Casen', 'Dilan', 'Summer', 'Sabrina', 'Kyro', 'Arturo', 'Rex', 'Zora', 'Albert', 'Ignacio', 'Raelyn', 'Aurelio', 'Juliette', 'Gabriel', 'Kamari', 'Rosie', 'Nolan', 'Rudy', 'Alessandro', 'Yareli', 'Alondra', 'Josie', 'Seven', 'Nayeli', 'River', 'Everleigh', 'Amelia', 'Miguel', 'Drew', 'Cal', 'Adler', 'Alvin', 'Issac', 'Julie', 'Sofia', 'Wade', 'Lillian', 'Westyn', 'Armani', 'Makenzie', 'Ares', 'Cassian', 'Harrison', 'Lilyana', 'Reyna', 'Elowyn', 'Lionel', 'Eve', 'Damir', 'Leyla', 'Maggie', 'Lucia', 'Jesus', 'Rocco', 'Reece', 'Lana', 'Jaylani', 'Amaris', 'Blaze', 'Kamiyah', 'Drake', 'Santino', 'Rosalyn', 'Derrick', 'Mariana', 'Colsen', 'Maximus', 'Zayne', 'Daphne', 'Loyal', 'Stevie', 'Zendaya', 'Javier', 'Rowen', 'Harold', 'Ira', 'Talia', 'Ismael', 'Johnathan', 'Truett', 'Shepard', 'Natalia', 'Karina', 'Estrella', 'Calliope', 'Wilder', 'Jayceon', 'Regina', 'Josue', 'Mordechai', 'Alina', 'Jamie', 'Dani', 'Magdalena', 'Monroe', 'Leanna', 'Ledger', 'Elijah', 'Maria', 'Cameron', 'Eli', 'Everly', 'Zaiden', 'Johnny', 'Zuri', 'Lia', 'Georgia', 'Jose', 'Darcy', 'Leo', 'Aspyn', 'Bailee', 'Santiago', 'Levi', 'Zaid', 'Icelynn', 'Jude', 'Miller', 'Kayla', 'Keilani', 'Maverick', 'Addilyn', 'Valeria', 'Sarahi', 'Matteo', 'Devin', 'Houston', 'Bella', 'Nikolas', 'Harlow', 'Westley', 'Cash', 'Micah', 'Dallas', 'Jayce', 'Mathias', 'Etta', 'Quentin', 'Chana', 'Shiloh', 'Kairo', 'Griffin', 'Lillie', 'Madelyn', 'Anaya', 'Ophelia', 'Kailani', 'Matthew', 'Elia', 'Aura', 'Piper', 'Jayden', 'Malcolm', 'King', 'Rory', 'Gunnar', 'Ermias', 'Bristol', 'Stetson', 'Elliana', 'Mohamed', 'Vivian', 'Caspian', 'Lylah', 'Bear', 'Aspen', 'Callum', 'Martin', 'Riggs', 'Lorelei', 'Aziel', 'Braelynn', 'Brayan', 'Rosalia', 'Nyomi', 'Ambrose', 'Nathaniel', 'Siena', 'Ozias', 'Estelle', 'Dior', 'Dafne', 'Imran', 'Jaden', 'Isabel', 'Cade', 'Emir', 'Sloan', 'Maia', 'Caroline', 'Wrenleigh', 'Liana', 'Nehemiah', 'Emilia', 'Ephraim', 'Quinn', 'Yasmin', 'Kaison', 'Nia', 'Edison', 'Esmeralda', 'Warren', 'Nasir', 'Kali', 'Avianna', 'Judah', 'Bianca', 'Alivia', 'Lettie', 'Pablo', 'Mateo', 'Ramona', 'Keyla', 'Rohan', 'Jovie', 'Nicolas', 'Sophie', 'Abdullah', 'Teresa', 'Makari', 'Darren', 'Serena', 'Kaya', 'Sevyn', 'Kendra', 'Dutton', 'Leroy', 'Makenna', 'Solomon', 'Moshe', 'Langston', 'Florence', 'Isaiah', 'Braxton', 'Lilliana', 'Wilson', 'Emanuel', 'Zaria', 'Yousef', 'Jack', 'Ashton', 'Khalil', 'London', 'Della', 'Nyla', 'Kaitlyn', 'Miracle', 'Kamden', 'Colby', 'Ailany', 'Danny', 'Salma', 'Kathryn', 'Alberto', 'Gunner', 'Ty', 'Oliver', 'Evangeline', 'Arianna', 'Mazie', 'Jake', 'Lyanna', 'August', 'Kannon', 'Grey', 'Clark', 'Casper', 'Brayden', 'Mariella', 'Cody', 'Philip', 'Bentley', 'Yael', 'Russell', 'Sylvia', 'Madeleine', 'Sophia', 'Emersyn', 'Akira', 'Legacy', 'Leah', 'Melody', 'Keaton', 'Johanna', 'Sterling', 'Ruthie', 'Aviana', 'Nellie', 'Jairo', 'Mckinley', 'Finn', 'Bruno', 'Jones', 'Love', 'Salvador', 'Cyrus', 'Duke', 'Cayson', 'Winona', 'Kaeli', 'Tyler', 'Juliana', 'Kendall', 'Kayson', 'Natasha', 'Jensen', 'Nicole', 'Yahir', 'Ryatt', 'Yisroel', 'Ximena', 'Layne', 'Aryan', 'Mae', 'Royal', 'Lorelai', 'Beau', 'Khaleesi', 'Goldie', 'Judson', 'Arthur', 'Rowdy', 'Isabela', 'Mallory', 'Roland', 'Trenton', 'Ellis', 'Tomas', 'Jeremias', 'Coleson', 'Faye', 'Sienna', 'Ryder', 'Daleyza', 'Cedric', 'Aron', 'Kyler', 'Romeo', 'Vera', 'Luke', 'Tilly', 'Irene', 'Ocean', 'Benson', 'Esther', 'Eugene', 'Xander', 'Fiona', 'Jrue', 'Kaylani', 'Jemma', 'Everett', 'Preston', 'Conor', 'Aylin', 'Trace', 'Bruce', 'Marie', 'Ruby', 'Zayd', 'Mac', 'Wesley', 'Damian', 'Jalen', 'Pedro', 'Khloe', 'Coraline', 'Tiana', 'Brady', 'Jonah', 'Kieran', 'Cain', 'Louis', 'Luis', 'Ayden', 'Alessandra', 'Hattie', 'Jon', 'Violette', 'Brynn', 'Crue', 'Chance', 'Case', 'Samira', 'Hannah', 'Mark', 'Idris', 'Ezra', 'Arisbeth', 'Jayla', 'Maya', 'Jasiah', 'Callahan', 'Ahmad', 'Jamir', 'Annabelle', 'Otto', 'Joel', 'Kylo', 'Briella', 'Conner', 'Mercy', 'Sadie', 'Sarai', 'Arlette', 'Vicente', 'Emery', 'Joe', 'Dream', 'Raylan', 'Adalee', 'Caiden', 'Manuel', 'Camilo', 'Elaina', 'Travis', 'Brinley', 'Royce', 'Brodie', 'Bryson', 'Anna', 'Spencer', 'Ryan', 'Pierce', 'Patrick', 'Paris', 'Viviana', 'Kiana', 'Hadley', 'Taylor', 'Kinley', 'Jaliyah', 'Trevor', 'Gracelyn', 'Adeline', 'Kinsley', 'Saylor', 'Izan', 'Chandler', 'Melina', 'Salvatore', 'Zariah', 'Amiyah', 'Celia', 'Amyra', 'Daniella', 'Brandon', 'Charli', 'Kayce', 'Violeta', 'Raul', 'Evander', 'Elaine', 'Paul', 'Zachariah', 'Emmie', 'Jett', 'Kolter', 'Margot', 'Eden', 'Ayla', 'Cooper', 'Lara', 'Colette', 'Frankie', 'Vanessa', 'Lane', 'Richard', 'Easton', 'Colt', 'Clara', 'Reuben', 'Chosen', 'Thiago', 'Koen', 'Novalee', 'Paula', 'Jessie', 'Eddie', 'Milena', 'Nathanael', 'Fabian', 'Juan', 'Cassius', 'Devon', 'Ayra', 'Reina', 'Nova', 'Jason', 'Alessia', 'Kate', 'Ailani', 'Ayleen', 'Lola', 'Nelson', 'Amias', 'Angel', 'Elisha', 'Gavin', 'Finley', 'Zeke', 'Alexis', 'Corey', 'Laith', 'Harlee', 'Corbin', 'Hugh', 'Zakai', 'Asa', 'Leilani', 'Amora', 'Santos', 'Kylee', 'Azaiah', 'Nash', 'Everest', 'Ariella', 'Selene', 'Jream', 'Damon', 'Camryn', 'Kaden', 'Winnie', 'Vincenzo', 'Kiera', 'Uriah', 'Michelle', 'Milo', 'Alisson', 'Shmuel', 'Renata', 'Harvey', 'Lukas', 'Neo', 'Leif', 'Marlowe', 'Elisabeth', 'Bjorn', 'Reagan', 'Brody', 'Elora', 'Marco', 'Angelica', 'Daniel', 'Zyla', 'Lance', 'Theodora', 'Denver', 'Lacey', 'Khalid', 'Ian', 'Kyree', 'Harlan', 'Colten', 'Alistair', 'Selena', 'Leslie', 'Lennon', 'Abel', 'Kenna', 'Kamryn', 'Theodore', 'Eduardo', 'Anders', 'Felicity', 'Arden', 'Allie', 'Wynter', 'Caden', 'Dakari', 'Remy', 'Braylon', 'Alaina', 'Chaim', 'Dante', 'Juniper', 'Amalia', 'Alia', 'Aadhya', 'Marlee', 'Sylvie', 'Kimber', 'Ishaan', 'Landon', 'Kade', 'Lily', 'Celeste', 'Evelyn', 'Adley', 'Cole', 'Andre', 'Barbara', 'Mackenzie', 'Penny', 'Axton', 'Mya', 'Salome', 'Aylani', 'Jazlyn', 'Jameson', 'Amayah', 'Aarav', 'Alfonso', 'Aliana', 'Kara', 'Millie', 'Alvaro', 'Romina', 'Journi', 'Livia', 'Christian', 'Ronan', 'Lawrence', 'Tatum', 'Virginia', 'Isaias', 'Marcellus', 'Miley', 'Sergio', 'Eva', 'Yusuf', 'Azriel', 'James', 'Oaklyn', 'Marigold', 'Pearl', 'Zaire', 'Juliet', 'Chaya', 'Bowen', 'Casey', 'Amoura', 'Mira', 'Kaysen', 'John', 'Kayleigh', 'Yusra', 'Kase', 'Kimberly', 'Laney', 'Alexia', 'Mia', 'Marshall', 'Ricardo', 'Thaddeus', 'Ari', 'Hassan', 'Birdie', 'Hudson', 'Hana', 'Alma', 'Finnley', 'Josephine', 'Boone', 'Adrianna', 'Kelly', 'Ezequiel', 'Colton', 'Erick', 'Henley', 'Alec', 'Flynn', 'Jaycee', 'Harper', 'Parker', 'Raymond', 'Arleth', 'Hugo', 'Ford', 'Peter', 'Haisley', 'Ainara', 'Abby', 'Milani', 'Abner', 'Brittany', 'Elias', 'Ariel', 'Grady', 'Amber', 'Opal', 'Soleil', 'Stefan', 'Mylo', 'Rylie', 'Alexandra', 'Aitana', 'Lian', 'Alfred', 'Rhodes', 'Lilith', 'Emiliano', 'Holland', 'Dulce', 'Celine', 'Helena', 'Alaya', 'Olivia', 'Steven', 'Meadow', 'Archie', 'Clayton', 'Jace', 'Oaklee', 'Enrique', 'Sariyah', 'Antonella', 'Palmer', 'Reed', 'Amir', 'Damari']


model_arg_dict = {
    'qwen2.5-32B': 'Qwen/Qwen2.5-32B-Instruct',
    'qwen3-32B': 'Qwen/Qwen3-32B',
    'llama3.3-70B': 'meta-llama/Llama-3.3-70B-Instruct',
    'llama3.1-DS': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'qwen2.5-DS': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'qwen_long': 'Qwen/QwenLong-L1-32B',
    'QwQ':         'Qwen/QwQ-32B',
}

chat_template_dict = {
    'qwen2.5-32B': './chat_templates/qwen2.5_chat_template_new.jinja',
    'qwen3-32B': './chat_templates/qwen3_chat_template.jinja',
    'qwen2.5-32B-int4': './chat_templates/qwen2.5_chat_template.jinja',
    'llama3.1-70B': './chat_templates/llama3.1_chat_template.jinja',
    'llama3.3-70B': './chat_templates/llama3.3_chat_template.jinja',
    'llama3.1-DS': './chat_templates/llama3.1_chat_template.jinja',
    'qwen2.5-DS': './chat_templates/qwen2.5_chat_template.jinja',
    'gemma3': './chat_templates/gemma3_chat_template.jinja',
    'phi4': './chat_templates/phi4_chat_template.jinja',
    'qwen_long': './chat_templates/qwen_long_chat_template.jinja',
    'QwQ': './chat_templates/QwQ_chat_template.jinja',
}


def writer_jsonl(filename):
    _exist = []
    if os.path.exists(filename):
        _exist = read_jsonl(filename)
    result_writer = open(filename, 'w')
    return _exist, result_writer


def read_file(filename: str):
    with open(filename, 'r+', encoding='utf-8') as f:
        a = f.readlines()
        # a = [line.decode('utf-8').strip() for line in f.readlines()]
    return a


def read_json(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        tmp = json.load(f)
    return tmp


def read_jsonl(filename: str):
    output = []
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)
            output.append(line)
    return output


def write_file(obj, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(obj)


def write_jsonl(obj, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in obj:
            f.write(json.dumps(line) + '\n')


def write_json(obj, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def reduced_multiplication(input_list):
    result = []
    for i in range(1, len(input_list) + 1):
        product = reduce(operator.mul, input_list[:i], 1)
        result.append(product)
    return result


def reduced_division(input_list):
    #c = reduced_division(b)
    result = [input_list[0]]
    for i in range(1, len(input_list)):
        result.append(operator.truediv(input_list[i], input_list[i-1]))
    return result


def random_partition(n):
    if n <= 0:
        return []

    # Generate random cut points
    cuts = sorted(random.sample(range(1, n + 1), random.randint(1, n // 2)))

    # Compute partition from differences between cuts
    partition = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [n - cuts[-1]]
    return partition


