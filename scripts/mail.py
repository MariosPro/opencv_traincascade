#!/usr/bin/python

import smtplib
import os
import yaml


class Postman:

    def __init__(self):

        self.load_credentials()
        self.smtpObj = smtplib.SMTP(self.credentials["server_name"],
                                    self.credentials["smtp_port"])
        self.smtpObj.ehlo()
        self.smtpObj.starttls()
        self.smtpObj.login(self.credentials["user_name"],
                           self.credentials["password"])

    def load_credentials(self):
        credentials_file = os.getenv("CREDENTIALS_FILE")
        if not credentials_file:
            credentials_file = "credentials.yml"
        doc = open(credentials_file, "r")
        self.credentials = yaml.safe_load(doc)
        doc.close()

    def send_mail(self, receivers, results):
        message = ("From: Pandora Victim Training"
                   + "<pandora_training@olympus.ee.auth.gr>"
                   + "\nTo: To Person"
                   + "\nSubject: Victim training Job"
                   + "\nTraining Result Message :\n")
        try:
            if isinstance(receivers, list):
                receivers = "".join(receivers)
            for key, value in results.iteritems():
                message += key + " : " + str(value) + " \n"
            print message
            body = "" + message + ""

            subject = "Training Result"
            headers = ["From: Pandora Victim Training",
                       "Subject: " + subject,
                       "To: " + receivers,
                       "MIME-Version: 1.0",
                       "Content-Type: text/html"]
            headers = "\r\n".join(headers)
            print body
            self.smtpObj.sendmail(self.credentials["user_name"], receivers,
                                  self.message)
            print "Successfully sent email"
        except:
            print "Error: unable to send email"
