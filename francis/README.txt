A. For the Item Feature Table, here are the columns: (from th MovieLens dataset)

*item_id
*movieId
*rating
*title
*genres
*year
optional:
* 1-hot encoding of genres


B. For the Transaction Table, here are the columns:
*user_id
*rating
*item_id


C. For the User Feature Table, its a count of the genres "seen" by the user (based on the genre classification of the movies. 1 movie can have several genres)

columns: 
*userId
*IMAX
*Adventure
*Mystery
*Animation
*Documentary
*Comedy
*Western
*War
*Film-Noir
*Crime
*Drama
*Thriller
*Fantasy
*Action
*Sci-Fi
*Children
*Romance
*Horror
*Musical
*(no genres listed)