# Privacy & Data Protection Notice

## ClassSight AI — Biometric Data Handling

ClassSight AI processes biometric information (face images and face embeddings) for student identity mapping in classroom environments.

---
 
 ## Phase 3: Behavior Analysis Ethical Guidelines & Privacy
 
 ClassSight AI Phase 3 adds behavior observation (phone usage, hand raise, head orientation, possible sleeping). The following principles strictly govern behavior analysis:
 
 1. **Observable Physical Indicators Only**: The system detects physical geometry (e.g. wrist position relative to shoulder, head orientation relative to body). It does NOT measure attention, interest, engagement, intelligence, or psychological state.
 2. **Neutral Labeling**: Observable physical descriptions are used instead of subjective judgements:
    - "Head-down for 8 seconds" — NOT "Student is bored / inattentive"
    - "POSSIBLE_SLEEPING" — NOT "STUDENT_IS_SLEEPING"
    - "Phone detected near student" — NOT "Student violating rules"
 3. **No Disciplinary Automation**: Behavior events are purely analytical and must never be used to automate penalties, grading deductions, or disciplinary actions.
 4. **Context Dependency**: A student looking down may be taking notes, reading a textbook, or working on an assigned worksheet. Camera angle and desk occlusion significantly affect detection accuracy.
 5. **No Psychological Profiling**: ClassSight AI explicitly disclaims any ability to determine mental health, cognitive ability, burnout, or emotional states from camera feeds.
 
 ---
 
 ## Phase 4: Engagement Analytics & Ethical Boundaries
 
 Phase 4 implements engagement scoring based on Phase 3 observations. The following boundaries apply:
 
 1. **Heuristic, Not Scientifically Validated**: The "Engagement Indicator" is an engineering heuristic, a configurable weighted formula combining observable events (like facing forward or raising hands). It is NOT a validated psychological or pedagogical measurement of true internal engagement, motivation, or learning outcome.
 2. **Absolute Transparency**: The scoring formula is completely transparent and explainable. Every score includes a breakdown of `contributing_factors`. We do not use opaque deep learning models for engagement.
 3. **No Student Ranking**: The system aggregates data to help teachers understand overall classroom dynamics. It does NOT rank, sort, or directly compare individual students against each other.
 4. **Protection Against Small Samples**: The `INSUFFICIENT_DATA` safeguard ensures students are not misjudged based on momentary observations or camera occlusion.
 5. **Opt-in Customization**: The heuristic weights are fully configurable by the educator to reflect the context of their specific lesson (e.g., heavily weighting "head-down" during a lecture, but removing the penalty during independent study time).
---
 
 ## Phase 5: Backend & Data Platform Protection Guidelines
 
 Phase 5 transforms analytical outputs into a persistent backend platform. The following data protection constraints are enforced:
 
 1. **No Biometric Data in PostgreSQL**: Raw face images and biometric face embeddings are strictly excluded from the relational database. Embeddings remain localized and are never exposed via REST APIs.
 2. **Role-Based Access Control (RBAC)**: Teacher accounts are isolated: an instructor can only view classrooms, lectures, attendance, and analytics for courses they manage. Administrator accounts are restricted to institutional administrators.
 3. **No Biometric Ingestion**: Ingested processing records contain only metadata, anonymized ByteTrack identifiers, confirmed student IDs, observable behavior events, and aggregate scores.
 4. **Attendance Safeguards**: Attendance is derived using configurable evidence thresholds (`MIN_ATTENDANCE_DURATION`, `MIN_ATTENDANCE_OBSERVATIONS`) to prevent false presence from momentary camera artifacts.
 5. **No Password Storage**: All passwords are cryptographically hashed using industry-standard bcrypt salt and hash algorithms. Access is managed via short-lived, signed JWT tokens.
 
 ---
 
 ## Contact

For privacy concerns related to this project, contact the project maintainer.

---

*This notice does not constitute legal advice. Consult qualified legal counsel for compliance with applicable privacy regulations.*

