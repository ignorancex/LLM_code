# Filename:     DNDO.py
# Description:  Dark Net Data Object (DNDO) - Like features across markets for ES indexing
# Version:      0.1 - Code Review Required before release
# Date:         June 29 2020
# Author:       Edward Crowder + Jay Lansiquot

class post:
    def __init__(self):
        self.title = None
        self.seller = None
        self.category = None
        self.creationDate = None
        self.url = None
        self.views = None
        self.purchases = None
        self.expire = None
        self.productClass = None
        self.originCountry = None
        self.shippingDestinations = None
        self.quantity = None
        self.payment = None
        self.price = None
        #Non scrapped data
        self.analyst_hasViewed = None
        self.analyst_viewDate = None
        self.analyst_flagged = None
        self.analyst_notes = None
        self.analyst_closedDate = None
        self.analyst_dateCollected = None

    def Post(self, title, seller, category,creationDate, url, views, purchases,expire, productClass, originCountry, shippingDestinations, quantity, price, payment, analyst_hasViewed, analyst_viewDate, analyst_flagged, analyst_notes, analyst_closedDate, analyst_dateCollected):
        self.title = title
        self.seller = seller
        self.category = category
        self.creationDate = creationDate
        self.url = url
        self.views = views
        self.purchases = purchases
        self.expire = expire
        self.productClass = productClass
        self.originCountry = originCountry
        self.shippingDestinations = shippingDestinations
        self.quantity = quantity
        self.price = price
        self.payment = payment
        #None scrapped data
        self.analyst_hasViewed
        self.analyst_flagged 
        self.analyst_notes
        self.analyst_viewDate
        self.analyst_closedDate
        self.analyst_dateCollected = analyst_dateCollected

    def toDict(self):
        print(self.__dict__)
        