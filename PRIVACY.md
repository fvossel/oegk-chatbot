# Privacy Policy — "Ask OEKG" Chatbot

_Last updated: 2026-07-21_

This policy explains how the **Ask OEKG** chatbot processes personal data. It
covers the chatbot application itself. For the Open Energy Platform website,
user accounts and the web server that hosts this application, the
[OEP Privacy Policy](https://openenergyplatform.org/legal/privacy_policy/)
applies in addition.

---

## 1. Identity of the controller

The controller responsible for the processing described here is:

- **Name:** Till Mossakowski
- **Address:** Universitätsplatz 2, 39126 Magdeburg, Germany
- **E-mail:** <glauer@iks.cs.ovgu.de>

If a data protection officer has been appointed for the controller's
institution, their contact details can be obtained via the address above.

---

## 2. What data is processed, why, on what legal basis, and for how long

### 2.1 Your chat input (questions)

**Data:** the free text you type into the chat, plus the last few turns of the
current conversation used as context.

**Purpose:** to translate your question into a SPARQL query, execute it against
the Open Energy Knowledge Graph, and summarise the result.

**Legal basis:** your consent, Art. 6 (1) (a) GDPR. You give this consent via
the checkbox shown before the chatbot becomes usable, and you can withdraw it at
any time (see section 5).

**Storage period:**

- **In memory:** the conversation is held in the server-side session for the
  duration of your session only and is limited to the most recent user turns.
  It is discarded when the session ends, when you reload the page, or when you
  withdraw consent.
- **In the request log:** each request is written to a rotating log file
  (`logs/requests.log`) with a timestamp, the **text of your question**, and
  technical metadata (whether a cached query was used, whether an error
  occurred, the number of result rows, and the processing duration). The log
  does **not** contain IP addresses, account names or any other identifier
  linking an entry to you. Entries are removed automatically once the file
  rotates.
- **In the error log:** warnings and exceptions are written to
  `logs/errors.log` and may incidentally contain fragments of a question. The
  same rotation applies.

The request log additionally records the moment consent is given or withdrawn,
together with the version of this document — again without any identifier. This
serves the controller's obligation to demonstrate that consent was obtained
(Art. 7 (1) GDPR).

> ⚠️ **Please do not enter personal, confidential or otherwise sensitive
> information into the chat.** The chatbot is a research and exploration tool
> for public, openly licensed energy data. It has no need for personal data and
> you should not supply any.

### 2.2 The shared question cache

**Data:** the text of a question, the SPARQL query generated for it, the time it
was created and last used, a usage counter, and a numeric embedding
(vector representation) of the question.

**Purpose:** to let any user re-run a previously answered question instantly
without another call to the language model.

**Important:** this cache is **shared between all users** of this instance. Its
entries are listed in the "Saved questions" sidebar and are therefore visible to
everyone using the chatbot. Nothing is written to it automatically — an entry is
only created when a user **explicitly** clicks "Cache this question" or adds one
through the manual form. Do not save a question that contains information you do
not want to publish.

**Legal basis:** your consent, Art. 6 (1) (a) GDPR, given by that explicit
action.

**Storage period:** until the entry is removed by an operator (the cache can be
cleared from the sidebar). Usage counters and the "last used" timestamp are
updated on every hit.

### 2.3 Technical access data

Requests to the web server (IP address, time, requested resource, browser
identification) are processed by the hosting infrastructure. This is described
in the [OEP Privacy Policy](https://openenergyplatform.org/legal/privacy_policy/),
which states a retention period of 14 days for server logs. The chatbot itself
does not read, store or evaluate your IP address.

The application does not set any tracking cookies of its own, and the
usage-statistics telemetry of the underlying Streamlit framework is switched
off in the application's configuration. Only the technically necessary session
cookie of the web framework is used, which is deleted when your session ends.

---

## 3. Recipients and transfers to third countries

### 3.1 OpenAI (language model and embeddings)

To generate a SPARQL query, to summarise a result and to explain a query, your
question, the relevant retrieved ontology excerpts and — where applicable — the
retrieved result rows are transmitted to the API of

**OpenAI Ireland Ltd.**, 1st Floor, The Liffey Trust Centre, 117–126 Sheriff
Street Upper, Dublin 1, D01 YC43, Ireland — with onward processing by
**OpenAI, L.L.C.**, United States.

This constitutes a **transfer to a third country (USA)**. OpenAI relies on the
EU Standard Contractual Clauses and, where applicable, the EU–US Data Privacy
Framework as transfer mechanism. According to OpenAI's
[API data usage policy](https://openai.com/policies/api-data-usage-policies),
data submitted through the API is **not used to train** its models and is
retained for a limited period for abuse monitoring only.

Because this transfer is unavoidable for the chatbot to function, and because
its risk cannot be excluded entirely, the chatbot is only made available after
your explicit consent (Art. 6 (1) (a), Art. 49 (1) (a) GDPR).

### 3.2 Open Energy Platform (knowledge graph and datasets)

The generated SPARQL query, and any subsequent request for dataset rows or
metadata, is sent to the public API of the Open Energy Platform
(`openenergyplatform.org`). These requests are authenticated with the
**application's own service token** — your identity is not transmitted, and the
platform receives only the query itself, not who asked it.

### 3.3 No other recipients

No data is sold, shared for advertising purposes, or passed to any other third
party. No web analytics, tracking or advertising services are used.

---

## 4. Rights of the data subject

You have the right, in relation to the controller named in section 1, to:

- **information** about the personal data processed about you (Art. 15 GDPR);
- **rectification** of inaccurate data (Art. 16 GDPR);
- **erasure** (Art. 17 GDPR);
- **restriction of processing** (Art. 18 GDPR);
- **data portability** (Art. 20 GDPR);
- **object** to processing (Art. 21 GDPR).

Please note that requests concerning chat input can only be fulfilled where the
data can still be identified. Because the request log deliberately contains no
identifier linking an entry to a person, it is generally **not possible** for
the controller to attribute a logged question to you (Art. 11 GDPR). If you wish
to have a specific entry of the shared question cache removed, please state the
question text in your request.

You also have the right to **lodge a complaint with a supervisory authority**
(Art. 77 GDPR). For the controller named above, this is the
Landesbeauftragter für den Datenschutz Sachsen-Anhalt.

---

## 5. Right to object and withdrawal of consent

Processing of your chat input is based on your consent. You may **withdraw that
consent at any time and with effect for the future**, without giving reasons and
without any disadvantage.

To withdraw it, use the **"Withdraw consent"** button in the chatbot's sidebar.
This ends the processing immediately, clears the current conversation from the
session, and returns you to the consent screen. Withdrawing does not affect the
lawfulness of processing carried out before the withdrawal.

To have entries removed from the shared question cache, or to exercise any other
right, contact the controller at the address in section 1.

---

## 6. Obligation to provide personal data

You are under **no obligation** to provide any personal data. Using the chatbot
does not require an account, and no field asks for personal data. If you decline
consent, the chatbot is simply not available to you; the underlying data remains
freely accessible on [openenergyplatform.org](https://openenergyplatform.org/).

---

## 7. Automated decision-making and profiling

The chatbot generates answers automatically using a language model. This
processing is **not** automated decision-making within the meaning of
Art. 22 GDPR: it produces no legal effect concerning you and does not similarly
significantly affect you. No profiles of users are created, and no behaviour is
evaluated across sessions.

Please note that the answers are generated by a statistical model and **may be
incomplete or incorrect**. They are not a substitute for the authoritative data
published on the Open Energy Platform.

---

## 8. Changes to this policy

This policy may be updated when the chatbot's processing changes. The
application detects changes to this document automatically and asks for renewed
consent when its content has been modified.
