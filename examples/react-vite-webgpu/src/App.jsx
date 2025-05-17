import { useEffect, useState, useRef } from "react";

import Chat from "./components/Chat";
import ArrowRightIcon from "./components/icons/ArrowRightIcon";
import StopIcon from "./components/icons/StopIcon";
import Progress from "./components/Progress";

const IS_WEBGPU_AVAILABLE = !!navigator.gpu;
const STICKY_SCROLL_THRESHOLD = 120;
const EXAMPLES = [
  "The report of the Civil Rights, Utilities, Economic Development and Arts Committee Agenda Item three Resolution 31669 Encouraging as a best practice the use of an individualized tenant assessment using the Fair Housing Act's discriminatory effect standards to avoid Fair Housing Act violations when criminal history is used as a screening criterion in the Landlord Screening Process, Committee recommends that the resolution be adopted as amended grade. I move to amend Resolution 31669 by substituting D four for version D three, which includes a new attachment. A And I understand Councilmember Bagshaw also has an amendment, but let's first, if we could, let me just go through the changes to the resolution since the last committee meeting. The changes are found in two recitals, as well as sections one and five are the primary changes. We added a recital that again lifts up the HUD guidance to show that a criminal history screening policy is next must serve a substantial, legitimate and nondiscriminatory interest. Another recital we included was referencing the the Seattle Fair Chance Employment Ordinance and the way they approach some of these same issues, looking at making sure that individualized assessments and prohibiting questions on initial job applications regarding an applicant's criminal record. And then in Section one, and these are changes that we worked on with stakeholders together with councilmembers, the wants office desiring to really focus on not the the impacts of this particular resolution, but for what we hope to see in the future ordinance that is going to be coming to us to to regulate this area of of housing screening practices. And it identifies the principles that came out of the Hallow recommendations. And in Section five, again, this is just clarifying that the expectation in the HUD guidance is to distinguish between criminal conduct, that that indicates a demonstrable risk to residents safety and conduct that does not the resolution itself, whereas it's really focused on encouraging landlords to to follow HUD guidance that has been recently released regarding criminal records. The separate sections do. A couple of different things. Sections one and two, again, focus specifically on the future legislation that we expect to be coming out of the mayors task force. The. The next section basically says that we endorse practices that landlords should not automatically exclude individuals for housing on the basis of prior event arrests. The further sections refer refer to the process that the Office of Housing has facilitated to create procedures to select tenant screening agency guidelines for property management and affordable housing. Another section recommends that the that a landlord should not rely on records that cannot be reported by consumer reporting agencies under state law. And another section focuses on the Office of Civil Rights efforts to basically do enforcement of existing fair housing laws through testing, investigation of charges and other other means. The final section requests that as OCR when determining whether or not a complaint of housing discrimination based on the use of criminal history, whether or not there should they should it ask them to seek to determine whether or not there's a disparate impact? So that's an overview of both the resolution and the the changes that have been made since the committee discussion and vote on June 3rd. And I don't I may have started talking before I had a second. May I add a second? All right, great. Those in favor of supporting the substitute version D for 4d3ii in a OC. So we have the substitute amendment before us. Councilmember Bagshaw will move to further amend the resolution, but before consideration of the amendment, we have to move to suspend the rules because we've got we received the text for the amendment after the , I believe, noon deadline today. So I moved to suspend Council Rule three A6 relating to presentation of full council amendments before 12 noon, checking all those in favor of the motion carries and we can proceed with consideration of the proposed amendment. Great. Thank you very much. What I am proposing is the addition in our whereas section two recognize what we all worked on a year ago called the Certificate of Restoration of Opportunity or the acronym was CROP. And it really and it's designed to address what the gentleman in the front row had talked about earlier today during public testimony , the state legislature passed unanimously out of both houses of the Act around Certificate of Restoration of opportunity. And what it is designed to do is to offer potential public and private employers or housing providers concrete and objective information about an individual who has served his or her time in prison and is now been released. And what we're really wanting to do here is to reintegrate individuals who have had a previous criminal history to provide housing and employment opportunities so that whereas that I am recommending we ensure it comes right out of the bill. And it would say that in an act relating to certificates of restoration of opportunity that will offer potential public and private employers or housing providers concrete and objective information about an individual under consideration for an opportunity. These certificates can facilitate the successful societal reintegration of individuals with a criminal history whose behavior demonstrates that they are taking responsibility for their past criminal conduct, pursuing a positive, law abiding future. So I'm just suggesting we add this, that it refers to this legislation, which I will hope a court will provide certificate, a restoration of opportunity, and an individual has something else in his or her hand to help him get a job or housing. So we have a first. We moved it and we second it as well. No. Okay. We have a move in a second. Now, all those in favor of the amendment to the resolution say I, I, I opposed me. And now we will have the full version before us to vote on any comments. Comment. Sorry, I have just some closing statements. I just really I think it's so important that landlords, housing providers in this community understand what the law is when it comes to developing policies and practices for making decisions based on criminal history. We know that we're not likely to have the ordinance that will do this work and until after the Mayors for Fair Chance Housing Committee will be reconvened in July, and they will have a series of meetings before they bring to us recommendations for for an ordinance. And so in the interim, it's really important that we lift up the the policies that HUD is currently currently promulgating and making sure that both landlords are engaged with the policy direction that the that the city is going to be pursuing in the future, as well as protecting themselves from fair housing complaints today. So with that those in favor of adopting the resolution vote i. I. Those oppose vote no. The motion carries the resolution is adopted and the chair will sign it. And now we can read items for through eight together.",
  "Madam Court, could you please read docket 1239? Certainly. Docket 1239. The Committee on Government Operations, to which was referred on December 1st, 2021, docket number 1239 message an order authorizing the creation of a sheltered market program in conformity with the requirements of general laws. Chapter 30 B Section 18. This authorization applies to contracts for goods, professional services and support services. This authorization is for no more than six contracts, which must be awarded by June 30th, 2022. This sheltered market program shall be available for disadvantaged, minority and women only vendors, for whom there is a demonstrated substantial disparity in the city's 2020 disparities. Study submits a report recommending the order ought to pass. Thank you so much, Madam Clerk. The Chair recognizes Councilor Edwards, chair of the committee. Councilor Edwards. You have the floor. This is this is actually a matter, I believe, sponsored by the. Mayor in Cannes. In conformance with the recommendations from the disparity study and making sure that we opt in to this this pilot program under mass general laws 30 Section 18. Again, it's really just following the recommendations of an already studied issue, which which demonstrates a disparity between minority contractors or women contractors receiving contracts in the city of Boston. So this would allow for us to shepherd and move these six contracts to those already designated groups who have a disadvantage. And I think it's. Really fulfilling a promise. Of making sure that we go through and make sure all aspects of the city government, including the financial benefits, are accessible to people in the city of Boston. I recommend that this pass and I hope that my colleagues will vote for it. Thank you. Thank you so much. Councilor Edward seeks acceptance of the committee report and passage of Docket 1239. Madam Court, could you please call the roll? Certainly. Docket 1239. Councilor Arroyo. Yes. Councilor Arroyo. Yes. Councilor Baker. Councilor Baker. Councilor. Councilor Barker. Council Braden. Councilor Braden. Councilor Campbell. Councilor Campbell. Yes. Councilor Edwards. Yes. Councilor Sabby. George. Councilor Sabby. George. He has Councilor Flaherty. Councilor Flaherty as Councilor Flynn. Councilor Flynn. Yes. Councilor Jane. Yes. Councilor Janey. As Councilor me here. Councilor me here as Councilor Murphy. Councilor Murphy. Yes. And Councilor O'Malley. Yes. Councilor O'Malley. Yes. Madam President, do I get number 1239 has received unanimous vote. Thank you so much. Dockett 1239 has passed and now we will move on to matters recently heard for possible action. Madam Clerk, if you could please read docket 0863. Certainly Docket 0863 order for hearing to discuss pest control and illegal dumping in the city of Boston.",
  "Item five, proposed ordinance 2016 0392. This is an ordinance relating to the transportation concurrency. And our our Transportation Department has done an amazing job of rewriting this, and they deserve a medal and a halo. So, Mr. Carlson, would you begin the briefing on 2006 0392? Thank you, Madam Chair. As you say, this proposed ordinance relates to the county's transportation concurrency program for the unincorporated area. It modifies the King County Code language on transportation concurrency, and it also approves a new concurrency travel shared boundary map and a new concurrency test results map. And for those who have not thought about concurrency in the past couple of years, that's a little bit complicated. So let me just give a brief outline. The and and I will say that Jay Osborne from Rhodes is here as well. And we have two members of the Transportation Concurrency Expert Review Panel. And Jay and I were planning to do the initial outline. First, there is concurrency language in the King County Comprehensive Plan, chiefly in the transportation chapter, and that sets requirements for the concurrency program. It also establishes the level of service standards for various land use areas. So for example, the rural area has a level of service for its roads, which traffic has to be more free flowing than in the urban areas. And the comp plan also requires that we do concurrency through the use of travel sheds and testing of traffic flow on arterials. I'm going to ask the committee assistant to call up our maps. That was the last handout so we can start. Under current law, we have 25 travel codes and you see the boundaries there are on the map. And in compliance with the comprehensive plan requirements, each of these travel sheds is an area where the traffic in that area uses the arterials and we test the travel speeds on the arterials. The code says every two years and then. The data is analyzed to generate a map showing travel sheds that are close to development because they fail concurrency. The way you fail concurrency is that. 15% or more of your miles on those arterials do not meet the relevant lower standard. So switch to the next. I guess I can switch to the next without. Or not. How do we get. How do we get to the next slide? Next one. Sorry. So we now have the Christchurch travel sheds are closed, there are five of the 25 and in each of those the roads and a little bit too much congestion during the afternoon peak. So the exceeded the standards and. The current proposed change in 392 makes a number of changes. It notably changes the travel shed boundaries. And if we go back to the last map, we see the new boundaries. It. Features of this map are new boundaries that reflect changes in the unincorporated area. It separates out the urban portions of the travel shields and makes them separate. And the urban unincorporated travel shelters are littered throughout. The new rural travel sheds are numbered, they are larger, and they reflect annexations. And they continue to have a logical configuration of roads that people under travel should use those arterials, and the arterial test is performed. It's not in the ordinance, but a new set of data from a local firm is used to identify the travel speeds on the arterials. And it's a much more thorough process of evaluating travel time than the old practice in which road staff actually got out and drove the roads and they had gizmos attached to their vehicles to monitor the travel times. So we have a much better picture of the actual travel times. Another change that is contained in this ordinance is under the current system, certain state routes are used in concurrency and the comp plan policy says that that may be done. It is at the discretion of the policymakers to use those routes. The the new proposal chooses not to use those state routes in the concurrency test and to stick with the county owned arterial routes. At this point, recognizing that this is a very complicated project, I'd like to suggest that Jay may come up. And if you have questions about what I've said, I've I've studied this a little too deeply, so explaining it is difficult for me. So with that, we do have a councilmember. Councilmember Dombroski has a question. Female Chair And Paul, thank you very much for your work on what seems like a little bit more of a new or a different approach to concurrency. On the last issue that you raised with the current plan, at least if I understand you correctly, permits us to include in the travel time analysis the use of state roads , which are an integral part of the world transportation system, for sure. What is the rationale for not using them? It seems like the average driver wouldn't necessarily distinguish between a state highway or state road and a county road when taking a trip. Okay, first it is certain state roads. It is not the statewide significant ones like the freeways. Right. The certain state roads that have characteristics similar to county arterials have a level of service standard that's set by. Is it the state or the Puget Sound Regional Council that, you know, the PUC sets them and so they're they are out of our control. And so the decision here was to focus on the roads that are within the county's jurisdiction and control. And in fact, the the data was gathered for the state routes that would have been used. And there is one shared that would have switched should two, but there wouldn't have actually been a difference in any other travel. Should switched from. What from open to to closed within this analysis. So in that case using the state roads because of their level of service standard will close, the travel said, meaning that development would be restricted. Now just saying no. So Jay Osborne, deputy director at roads. So the state highways in question two or two, two or three of our 900. This level of service for those is D in the rural area, but the counties level of service for roads in the rural area is B So when we test the state routes against the counties level of B, they do not pass, but they meet the state level of D and our passing the concurrency test for the state's purposes. So one of the complications in what we've been doing is to test state routes at our standard and not the states, whereas they don't meet our standard of B, they are meeting the state standard and therefore passing concurrency, which has been one of the mixes and testing state routes in the rural area. So what is the implication of this policy choice to somebody who has property and wants to do a plot to build homes. In the rural area? At the moment, using the test that we proposed, it would pass and they would have the ability to do that. And for example I've been travel said to. Councilmember Lambert's district mostly there. If you. He used the state standards and roads. He'd have like two or two up there. Kathy is that right? Mm hmm. And it might close the travel show, right? If we use the county stamp. Staff applied the county to the state. Yes, exactly. Okay. Thank you, Madam Chair. Thank you for your patience as I kind of worked through that. Yes. Councilmember, you're going to speak in detail. So but let me just put my question on the table so you can think about it. So this councilmember is used to judging concurrency at intersections. So this is sort of new to me. So it would be really helpful if you could kind of lay out how it's calculated and how you did it. But if we could just sort of reinforce that also what causes failures is some kind of average of all the roads are because if any one road is failing, that doesn't mean the whole travel shed is failing . And then just the bottom line here is if that particular change you were just discussing is made, I think the net real world impact is we're going to allow more congestion and continue to allow development. Okay. So which is I mean, which is it's about a reasonable reaction to a real world situation. But essentially, we're going to allow we're going to lower our standards for how free flowing the traffic has to be in order for development of energy to be allowed. Yeah. So there are many ways to test concurrency. So the counties methodology, we used to do actual travel time, which meant that we had staff in cars with stopwatches in the nineties driving a length of all of novelty road with a stopwatch to see how long it would take. Going back and forth between three and 6 p.m. three times in the spring when school was not was in session and not at spring break. And we would have people standing on overpasses with the stopwatch. So that's the difference. And we do the whole length of the roads. And so it's the arterial roads in that sched are all tested. And the rule is 85% of them have to meet your standard for the shed to pass concurrency. When you add state routes in because they exist in those areas and you test them at the counties level because state routes are designed to take a greater level of traffic than some of the arterial roads that we have, they don't meet the level of service. B We had a group of grad students from the U. Dub who did their MBA thesis on concurrency and found that the counties level of service B is one of the highest in the country. That set as a very aspirational level of service. So the state choosing to put D on those routes and then passing is a different standard and testing those creates complications. So to answer your question, if we added the state route testing in which we did do, and to back up for a moment about what we're doing with data, there is a firm that we were able to buy data for 24, seven for a month on those roads and then pull out the testing for every day Monday through Friday. Actually, I think we use Tuesday, Wednesday, Thursday data for those afternoons. So rather than someone driving three or six times, we had all of those data points to test out how the traffic was running with this index data. Yes. So based on cell phones. Yes. And pulling that together, which was actually more cost effective than paying staff to be out in cars on the road, testing all those areas that and we don't have that many staff to left to drive on those roads in our planning group. So one of the things that in the rural area I think is key is that letting the zoning code deal with development and how much development is actually left in the rural area is very small for the impacts on the county road system and how much development is going to put cars on the road in those open sheds. And the impact there in. Terms of climate should be thought. And this is a bigger than just the county in dealing with unincorporated areas. I would dearly love it if we could come up with some kind of a regionally consistent way of doing concurrency. And I could see it being done differently in rural areas and urban areas. But the way the city of Seattle does it, to the extent they do it at all, the way that cities like Bellevue do it and the way you all do it are all completely different. And so it's very hard for us to have a common vocabulary for the public around how well or badly the roads are doing . And then we end up essentially being driven in transportation policy by individual anecdotes of my experience behind the wheel. And I think that that's important how people are feeling about their commutes. But it would be much, much better if we had a systematic and clear way of talking about what's going on with the whole system. That's my soapbox on these things. I also dearly wish that we had a way of including throughput as part of our calculations, because it's not just how fast the individual vehicles are moving that's important. It's how many people were able to move through, you know, these points from point A to point B, if you have a really a an arterial that's really well served by transit, even if it's going slower, it could be moving lots more people than one that is really not well-served by transit. So anyway, thank you for listening. Maybe next time we can work with our colleagues on trying to bring some of this stuff together, but I appreciate the work you've done here. I think it makes good sense given the realities of what we're dealing with, especially splitting out the urban from the rural makes good sense. Thank you. So, Councilmember, if you had been here years ago when we had our old currency plan that had, what, 300 and how many, 360 boxes? It was a nightmare. And we hired a national firm to come in to to give us some feedback. And there was the worst plan they'd ever seen in the country. This man said he had like 30 years experience and he'd never seen anything like it. So he pretty much ditched that afterwards. And so this is the new improved, the new improved, the new improved. And we don't have any jurisdiction, as you said, over Bellevue now that you're not mayor anymore. So we really can't unless it goes through, you know, your transportation committee. He has RC to, you know, to make those changes. But I think the thing that's really important is that people are driving from the rural area into the unincorporated incorporated areas and there's one level of service out here and then there's another level of service in here. The drivers driving, they know they're going to be in commute traffic. And so to have one level of service that is asked for, okay, that is aspirational so that, you know, it's an artificial barrier. And I don't know if it's. Thank you for handling that out there. I don't know. Is it still true that on the urban growth boundary line that the the part of the road from the Senate line to the rural area has one level of service and from the center line to the urban line is another level service, or do we fix that? A couple of years ago, I can't remember if we did or not. We did fixed fixed it. Okay. So that used to be a problem that the same road could have two levels of service on it. So we did fix that. So that's good. So I think that this makes it easier. It's more consistent with other roads in the county. And the other point I think is really important is that there isn't much development left in this county other than what's already been delineated under the Growth Management Act. So we know what that's going to be. So I think this, as Jay said, is an ability to deal with the roads that we have control over. And if I may make one final thing, thank you. And then I promise that will be be. And when I look at this map and I see the circles inside the travel sheds, those are the urban islands in the unincorporated areas. I think there's been a lot of talk and consternation about the growth targets in those areas. And I have to say that this map demonstrates part of the reason why there's a debate and why it isn't just a one sided. We need to grow. We're growing. Let us grow. The other side of that coin is the more we allow or encourage or support large amounts of growth out in those urban areas, the more you're going to see these travel shed suffer because they have to serve and and support growth between there and end. We are still requiring a more free flowing state of traffic for there to be ongoing development than we are in the inner suburbs of the urban areas. So it's a this sort of demonstrates one of the complexities of that whole debate. And, you know, it gets into a lot of the debates with we're talking about affordable housing when we're talking about, you know, certain kinds of lifestyles in the rural area where the cities, which is where people are supposed to be, you know, growing out there. And it's a housing choice that for some parts of the county, there are very different housing choices in different parts of the county. So it is a complicated issue indeed. Okay. Did you want to continue? Well, I was going to say on page 46 of your packet is the actual list of the root segments that failed in this analysis. And those are miniatures which travel should they're in, as you said, the total mileage within the travel shed. If 85% or more passes, that's the test. If less than 85% of the mileage fails, then the travel should is failing and only one travel should fails in this new process. The the other point that Jay alluded to in terms of development in the rural areas, even if the travel should fail, there are there is provision for minor and certain public and certain educational developments to proceed. And our concurrency system has always allowed that the form in which that has been authorized has changed. And so a section of the code that is amended in this proposed ordinance for. Ten 7285. Lists those minor developments and schools and other uses that can still go forward if a travel should fails. And that's particularly important, for example, because one of the old issues that we heard a lot about was a family that had owned a parcel that wanted to subdivide so the children could build a house. And that's something that the county has modified the program to accommodate that kind of use as long as it complies with the zoning. And, you know, again, concurrency is the first step in developing something. You have to be consistent with your zoning as well. So I think we should show the last map which shows the results. That's the point. Yeah. This is so this is the test results map. And the red arrow there shows the one close development the close showed, which is mostly agricultural production district and is it does not have a lot of areas that could be developed anyway . Oh, that's interesting. It is APD and the parcels, there are probably minimum ten acres, so you're not going to be getting a whole lot of traffic out there. So how did that end up getting closed? It's okay. On page 46, that's seven and there are two. It's a small shed with a small mileage and there are two road segments, each a half mile long that fail. And that puts it over the top. It's. It's an odd area because of the agricultural uses. So to 72nd to 77, that's the main drag across the valley goes through the APD, which is four lanes going through there at an urban level and it is being tested at a level of service fee because the ag area is rural. So it's the urban road going through there, being tested at the rural level, which is why it fails. So if it's an urban road, why are we testing it, the rural area, just because it's in the APD? All right. Okay. So what? Oh, council member and about. Sorry. Thanks. I think when we looked at the concurrency issues a year or so ago, we kind of parked them because there were some open issues and the testing had gone and done this work. And one of the issues at the time and this kind of falls on concern about duties, regional consistency, if you will, but on a more narrow basis. And that was I was interested in travel sheds that cross the urban rural line. And at the time we had something like up the East Renton Plateau and and there was then a question about whose standard should you apply? Right. And it seemed to me it made sense to at least take a look at the adjacent city standard and the urban side of the line along the line of thinking that we should account for, you know, the city's planning policies and zoning traffic standards, that kind of thing. So my one takeaway that I'm getting from this, I'd like to make sure everything is it's a move to not cross the urban rural line in travel sheds. We've now got travel sheds on the rail centerline line where we can have one set of standards and then on the urban center line where a different step might have apply some nods there. That's that looks good to me. And the related question then is within the urban side of the line, will we do we in this proposal or will we in the future start taking a look at the city, you know, the city that has the paid for their standards and incorporating that into our into our level of service standards. So in the history of concurrency, we've had agreements with various cities to do that and to do that development. When the economy suffered in 2008, there were four cities that withdrew from those agreements looking to be able to develop their areas and what they needed to deal with, because there's also an impact to the mitigation payment system and how much money that you were getting for development. It proved to be somewhat complicated as they went forward. And we have had conversations with some cities about those standards and those areas continue to annex Kahani and some Amish being an example and is acquire Fall City Road, some issues that they were interested in developing. But currently we don't have any agreements to model concurrency in those areas, in part because the remaining urban areas are quite small now and have been chipped away at. So I think it's important to know that we as a county in the state have the smallest amount of unincorporated area of any of the county. We have 12 and a half percent of the county that's unincorporated. Snohomish County has 28% and everybody else is in the forties and above. So we have done what the Growth Management Act said and again Incorporated. So I'm talking about 12%. Yeah, I'm talking in rural. Urban or rural. Unincorporated. And I'm just looking at the map here. I don't think I don't think 12% of the county is. An anchor, but. I think at least landmass, it's it's quite a bit bigger, maybe population, it's probably about 300,000 out of two. It's about 250 something. So anyway, that and I don't remember exactly how they calculated that 12 and a half percent, but we are far lower than everybody else. And so the land has already been allocated to whatever it is, one acre, five acres, ten acres, 20 acre parcels. So I think that as we go forward and we look at that, we have an aspirational level of B and then the people get off the B road and they get into a city and it may have a D or an F rating. The dichotomy of being and on this part and going this half mile at a B and then this half mile at at F or this half mile, the D, I think that as as we look at this, we need to be more realistic about how high that level is and making sure that, you know, people who own land or would like to have their properties, the device, the children could live on the property and take care of them. That. We make allocations for that. So. Okay. So what is the the will of the body? We need to have a 30 day period for putting this out for when they call it public testimony. 30 day advertising period. So would you like to vote this out of committee with or without recommendation? What would you like to do? Well, there's an amendment that has that been described as looking like mostly technical cleanup. Or is there some policy changes there? Well, yes and no. Yes. Yes. Okay. Yeah. You have before you Amendment one a which is very slightly different from the amendment in your packet. It I would say that, yes, it makes mostly technical changes. There were a couple of spelling errors. There is a new sentence added to section eight. The first. And this is. How does that. No, no, we're not at all yet. How did it get up to be, anyway? It went in front of me. Yeah. The section four of the ordinance there is the online nine amendment when it says except except as provided in KCC 1470 285. That's I would call that a technical clarification that a minor use is covered by 285, which is which has always been the case. So it's not changing any practice when ten is a typographical change. Then starting on the line 13 Section 1470 285 L This is the last item on the list of those minor developments that are allowed in feeling travel sheds. And this is there's some rewording for clarity. And then down on line 18, it says the property has not been subdivided in the last ten years. This relates to a short subdivision in a rural travel shed where the owner wants to subdivide. And this is the classic family method of requests. And under current law, if the applicant has owned the property for five or more years and the property has not been subdivided in the last ten years, then that's allowed if it meets the zoning requirements and there is no need to purchase transferrable development rights as part of that deal. This is this is how it has been. The executive transmitted proposal was going to change the no subdivision in ten years requirement to no subdivision in five years. And in reviewing this, we found that there is a rural policy are 3 to 3 in the just approved plan that says ten years is the requirement. So we're we are maintaining the existing language for ten years and not moving forward with the change that the executive has proposed. And it is simply because comprehensive policy language is mandatory on that point. Can I ask the question? Is the executive okay with the revised amendment? Yes. Okay. So I would have preferred the old, but the new was what we just passed out. So I will tell you that when the plan comes up again in four years, that I would like to reconsider this. But but anyway, that's the way it is at this point. Councilmember Balducci. No, my questions were answered. Thank you. Okay. Councilmember Dunn. Thank you, Madam Chair. Just a concurrences, an issue I've worked on for a long, long time. What? What? You know, it's fairly not well understood by most elected officials. I fear you're changing the slightly modifying and expanding the travel sheds, but you're not changing the methodology methodology for the actual concurrency standards in this. Right. Is that correct? The the methodology the in the in the service. Level, E for example, those. Sorts of things. You know, the the comprehensive plan establishes the level of service standards for urban as E, rural as B and then there are the rural town centers are D and rural no are E and rural neighborhood centers are D. You did not change those level of service standards in this latest update of the comp plan. So they still remain in place. Okay. And and we used to use a red, yellow and green map for concurrence. You remember that? And that's gone. Is that no longer what we're using? We're going to modify to this this mapping. That that was when when we had those hundreds of individual zones. And at one point it was written red and green only that it was red, yellow and green. When we moved to the Travel Shed concept a few years ago, 2008, I think it was the colors were abandoned. Okay. You know, I generally, maybe more than most up here, I tend to believe we need to. Be building homes. Condos, low income housing, what have you, because we need to put places for young families to live and for everyone to live. And so I'm with what you might call pro-development, but we've got a situation that's developing in earmuffs. Right? I know pro-development. I know it's bad. I didn't win. So. Yeah. And and so the question I have, I'm looking at south of Issaquah, you know, the is for Hobart Road, a road that is so bad that I pretend it doesn't exist because you will get lost in the vortex of traffic forever and they are deeply unhappy citizens there. People can't get in and out. Emergency services can't get in and out. Ambulances, fire trucks. It's it's awful, largely because this county refuses to increase capacity on the road. That's another issue. But I'm not seeing something here that's precluded development in that travel shed. What's the status of the Esquire open road travel should I think I saw was number 12. Well, no, you're on spot here. So if you want to pass it on to some of your colleagues. Well, the the crude travel schedules for the new travel shed would also be open and. There is a segment of Issaquah Hobart Road that feels it's between the Issaquah City Limits and Southeast 127th Street. So. So there wouldn't be development wouldn't be permitted to make a long story short in that section. No, it's it's the total results for the travel should in within within travel should for you do not hit the 15% or more mileage feeling standards. Okay, that's it. So I'm almost done. Madam Chair, I appreciate that. Okay. So I've never believed that currency ought to be the way to control our land use planning and development. I think that's the wrong way to do it. I think we ought to be doing it through zoning, through other permitting related issues. But we've got a. Real problem with this for Hobart Road. Part of it is a willingness to increase capacity. A bigger part of it probably is the fact we have the money to increase capacity. Maybe it's a little bit of both, but I just want to point out that. If you put large developments, even if they're an RFI of zoning out there, you are just going to add to a problem that is already disastrous. It's more of a statement than a question. And so. Makes me wonder whether these broader travel sheds are really the right way to go. As a matter of policy, I'm not going to object against it, but I'd like to drill down on it further in the future. So since I've been here, we've had several different renditions of what the concurrency looks like, and it's gone from absolutely obnoxious to be figured out. You need like a Ph.D., which is what the expert came in and said to something more simplified. The amount of growth going on out there is is very small compared to what it used to be. So I want to clarify. I've gotten some clarification. 12% of the population in this county is in the unincorporated area. So it is by population 12% of the population. Half of that is in areas that can be annexed. And so 6% is in the rural unincorporated. So there's not a lot out there. So we have some people with us that need to comment too and had some really important things to do that have studied this. So do you want to make some comments also? So just as a quick introduction, okay. In this, the council a number of years ago appointed a Transportation Concurrency Expert Review panel to review the work and provide a comment letter on every thing that was submitted to the council as we went. And it's represented from folks from the development community, the environmental community. We have a citizen of the unincorporated area and we have a representative of the Non-Motorized users and bus and transit as well on that. In this legislation, the Transportation Concurrency Expert Review Panel has decided that their time has come to an end and that the methodology and the amount of development and what we're doing with concurrency is something that they support. And the 1:00 scholar who's to my right, who is the chair of the Transportation Concurrency Expert Review Panel, it's going to give you a few remarks. Good morning, council members and thank you very much for your time. I would also like to let you know that one of our very long standing members and Martin is here in the back and I think her attendance, in addition to mine, I hope, conveys a reflection that this panel had quite a committed and long standing involvement with staff as both historically in terms of the older approaches to concurrency as well as the approach that we're putting forth to you now in these materials. Probably the most significant aspect of the panel that I, I was influenced by was both the collaboration among extremely diverse interests, as well as what we're giving to you today, which is essentially a unanimous recommendation, despite the broad diversity of interests on the panel. This panel has worked together for many years. I am the most recent addition to the panel, although I, through my former former colleague Bob Jones, knew a lot about what was going on in the background and am extremely honored to have taken over chairmanship of the panel a couple of years ago. And we're pretty proud of well, I should say we're extremely proud of the work that our are very well-educated staff has done on this, as well as the master's program materials that were presented to us in the past year. One of the things that I think is pretty interesting is that staff has been pretty selfless in this process. They were very interested in the good of the county and the good of the system above all. And in looking at that data, I think that is really reflective of some very thoughtful work that's been given to you . So and I would say that all of us on the panel again felt that we were extremely well served, not just by our consensus, but by a very well-educated group of individuals who could really so succinctly convey information to us to allow us to have a pretty candid and often very spirited dialog, as you might imagine, considering the members on the panel. But ultimately, we feel good about very good about what we've presented. We're sad about dissolving because it's one of those few fora where we actually get to get together and talk candidly without having to put other people's interests on the line, but really have good quality conversations. But it makes a lot of sense at this point to dissolve. And so we're very honored to have served the county. We thank you very much for the opportunity, and I hope that we can continue to be of service in our individual capacities. First of all, I'd like to thank you, as it's been said a number of times, if here this is a very complicated formula, it's very impactful. And so having somebody willing to sit down and look at all this and and bring a unanimous decision back is very much appreciated. And we thank you for your service. So essentially what I'm hearing you say is that you believe some tell me this is right, that you believe that with the lack of growth happening right now, that there's no need for you to continue on as a committee to evaluate this. The panel believes that both because of the way that the travel sheds have been reformatted and the annexation processes that are going on, as well as the ability to use a lot of that more mechanized methodology through INRIX, that there just isn't a need for this panel to both take their time to review these aspects that, yes , are becoming a little bit more rote in their processing. And we don't need to take staff time to be putting together materials when we don't necessarily have a deliverable we may need to bring to you. I don't know what the future holds, but for the moment, we're comfortable with the decision. Excellent. Okay. And I'd also like to thank you for being so cognizant of other assets like INRIX. And I know we use it at other committees and the data has been very, very helpful. So thank you for seeking that out to you. Okay. So now I'd like somebody to put this before US Council member. But did you manage to move approval of proposed ordinance number 2016? Dash 0392 of the do pass recommendation. Thank you, ma'am. Any questions or comments before we take the vote? Okay. Councilmember and Ambassador. I'd like to offer Amendment one. Oh, yes. Thank you very much. Yeah, I did have a question before we and we can vote on that, but just got a final. Okay. Thank you. I think this has been well explained by our staff. It isn't exact, as my name's on it. When I'm speaking to it, I would prefer that some of this wasn't changed this way. The correctional errors and the typos and stuff. That's great in the clarifications in the King County code, that's fine. It is the five and ten year issue that that does bother me. But because we just passed the comp plan that was voluminous and somehow that was the change in there, I think we need to flag that for three years and ten months from now and maybe change it back. But at this point, having all of our code be consistent is probably a good thing. So all those in favor of when a as presented by our staff please say I as opposed name is passed and now before us we have the amended version of 2016 0392. And Council Member Dombroski has the comment. Just to make this a follow up to Councilmember Dunn's inquiry about capacity and related funding. And when somebody does a project and they may pay some mitigation money, right? Does that money under our current provision, does it need to be spent within the travel shed where the projects occurring? So in the current provisions, it's SIPA money that they're actually spending on the roadway. And so it's for specific projects and identified for those within the travel. Within the travel said, okay, thank you, thank you, thank you. Okay, thank you. That was a good clarification question. I'm glad you asked that. Okay. Are those in favor of call for the vote from the clerk's office? Councilmember Baldacci. Councilmember Then back. Councilmember. Then I remember. Gossage. Councilmember Colwell. Councilmember McDermott. Councilmember of the group. All right. That's number one right there. Madam Chair, I mean, I'm sure the vote is six days, zero no's and councilmembers, Gossett, McDermott and moderate. They were excused. Okay. So. Do we want this on consent or do we want to talk about it again? What would you like? Didn't I hear you say that it needs to be put out for Thursday? Public comment. Oh, that's right. That's right. Says that's not enough for sure. Okay, that's good. Thanks for pointing that out. Okay. And it does take a 30 day advertising period which can start. So this will not be on the regular schedule because we have to wait for the after the 30 days, which will be the end of February. So if there is no other business to come before this committee, the meeting is adjourned. Thank you, everybody.",
];

function App() {
  // Create a reference to the worker object.
  const worker = useRef(null);

  const textareaRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Model loading and progress
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [progressItems, setProgressItems] = useState([]);
  const [isRunning, setIsRunning] = useState(false);

  // Inputs and outputs
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [tps, setTps] = useState(null);
  const [numTokens, setNumTokens] = useState(null);

  function onEnter(message) {
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setTps(null);
    setIsRunning(true);
    setInput("");
  }

  function onInterrupt() {
    // NOTE: We do not set isRunning to false here because the worker
    // will send a 'complete' message when it is done.
    worker.current.postMessage({ type: "interrupt" });
  }

  useEffect(() => {
    resizeInput();
  }, [input]);

  function resizeInput() {
    if (!textareaRef.current) return;

    const target = textareaRef.current;
    target.style.height = "auto";
    const newHeight = Math.min(Math.max(target.scrollHeight, 24), 200);
    target.style.height = `${newHeight}px`;
  }

  // We use the `useEffect` hook to setup the worker as soon as the `App` component is mounted.
  useEffect(() => {
    // Create the worker if it does not yet exist.
    if (!worker.current) {
      worker.current = new Worker(new URL("./prompt-compressor.worker.js", import.meta.url), {
        type: "module",
      });
      worker.current.postMessage({ type: "check" }); // Do a feature check
    }

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "loading":
          // Model file start load: add a new progress item to the list.
          setStatus("loading");
          setLoadingMessage(e.data.data);
          break;

        case "initiate":
          setProgressItems((prev) => [...prev, e.data]);
          break;

        case "progress":
          // Model file progress: update one of the progress items.
          setProgressItems((prev) =>
            prev.map((item) => {
              if (item.file === e.data.file) {
                return { ...item, ...e.data };
              }
              return item;
            }),
          );
          break;

        case "done":
          // Model file loaded: remove the progress item from the list.
          setProgressItems((prev) =>
            prev.filter((item) => item.file !== e.data.file),
          );
          break;

        case "ready":
          // Pipeline ready: the worker is ready to accept messages.
          setStatus("ready");
          break;

        case "start":
          {
            // Start generation
            setMessages((prev) => [
              ...prev,
              { role: "assistant", content: "" },
            ]);
          }
          break;

        case "update":
          {
            // Generation update: update the output text.
            // Parse messages
            const { output, tps, numTokens } = e.data;
            setTps(tps);
            setNumTokens(numTokens);
            setMessages((prev) => {
              const cloned = [...prev];
              const last = cloned.at(-1);
              cloned[cloned.length - 1] = {
                ...last,
                content: last.content + output,
              };
              return cloned;
            });
          }
          break;

        case "complete":
          // Generation complete: re-enable the "Generate" button
          setIsRunning(false);
          break;

        case "error":
          setError(e.data.data);
          break;
      }
    };

    const onErrorReceived = (e) => {
      console.error("Worker error:", e);
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener("message", onMessageReceived);
    worker.current.addEventListener("error", onErrorReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => {
      worker.current.removeEventListener("message", onMessageReceived);
      worker.current.removeEventListener("error", onErrorReceived);
    };
  }, []);

  // Send the messages to the worker thread whenever the `messages` state changes.
  useEffect(() => {
    if (messages.filter((x) => x.role === "user").length === 0) {
      // No user messages yet: do nothing.
      return;
    }
    if (messages.at(-1).role === "assistant") {
      // Do not update if the last message is from the assistant
      return;
    }
    setTps(null);
    worker.current.postMessage({ type: "generate", data: messages });
  }, [messages, isRunning]);

  useEffect(() => {
    if (!chatContainerRef.current || !isRunning) return;
    const element = chatContainerRef.current;
    if (
      element.scrollHeight - element.scrollTop - element.clientHeight <
      STICKY_SCROLL_THRESHOLD
    ) {
      element.scrollTop = element.scrollHeight;
    }
  }, [messages, isRunning]);

  return IS_WEBGPU_AVAILABLE ? (
    <div className="flex flex-col h-screen mx-auto items justify-end text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900">
      {status === null && messages.length === 0 && (
        <div className="h-full overflow-auto scrollbar-thin flex justify-center items-center flex-col relative">
          <div className="flex flex-col items-center mb-1 max-w-[500px] text-center">
            <h1 className="text-4xl font-bold mb-1">LLMLingua-2 <br />in your Web Browser!</h1>
          </div>

          <div className="flex flex-col items-center px-4">
            <p className="max-w-[514px]">
              Compress any text into shorter text!
            </p>
            <p className="max-w-[514px] mb-4">
              Check out the source code for this demo on{" "}
              <a
                href="https://github.com/atjsh/llmlingua-2-js"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                GitHub
              </a>.
            </p>

            {error && (
              <div className="text-red-500 text-center mb-2">
                <p className="mb-1">
                  Unable to load model due to the following error:
                </p>
                <p className="text-sm">{error}</p>
              </div>
            )}

            <button
              className="border px-4 py-2 rounded-lg bg-blue-700 text-white hover:bg-blue-900 disabled:bg-blue-100 disabled:cursor-not-allowed select-none"
              onClick={() => {
                worker.current.postMessage({ type: "load" });
                setStatus("loading");
              }}
              disabled={status !== null || error !== null}
            >
              Load model
            </button>
          </div>
        </div>
      )}
      {status === "loading" && (
        <>
          <div className="w-full max-w-[500px] text-left mx-auto p-4 bottom-0 mt-auto">
            <p className="text-center mb-1">{loadingMessage}</p>
            {progressItems.map(({ file, progress, total }, i) => (
              <Progress
                key={i}
                text={file}
                percentage={progress}
                total={total}
              />
            ))}
          </div>
        </>
      )}

      {status === "ready" && (
        <div
          ref={chatContainerRef}
          className="overflow-y-auto scrollbar-thin w-full flex flex-col items-center h-full"
        >
          <Chat messages={messages} />
          {messages.length === 0 && (
            <div>
              {EXAMPLES.map((msg, i) => (
                <div
                  key={i}
                  className="m-1 border dark:border-gray-600 rounded-md p-2 bg-gray-100 dark:bg-gray-700 cursor-pointer"
                  onClick={() => onEnter(msg)}
                >
                  <div className="font-bold">
                    length: {msg.length.toLocaleString()}
                  </div>
                  <div className="text-sm">
                    {msg.slice(0, 200) + "..."}
                  </div>
                </div>
              ))}
            </div>
          )}
          <p className="text-center text-sm min-h-6 text-gray-500 dark:text-gray-300">
            {tps && messages.length > 0 && (
              <>
                {!isRunning && (
                  <span>
                    Generated {numTokens} tokens in{" "}
                    {(numTokens / tps).toFixed(2)} seconds&nbsp;&#40;
                  </span>
                )}
                {
                  <>
                    <span className="font-medium text-center mr-1 text-black dark:text-white">
                      {tps.toFixed(2)}
                    </span>
                    <span className="text-gray-500 dark:text-gray-300">
                      tokens/second
                    </span>
                  </>
                }
                {!isRunning && (
                  <>
                    <span className="mr-1">&#41;.</span>
                    <span
                      className="underline cursor-pointer"
                      onClick={() => {
                        worker.current.postMessage({ type: "reset" });
                        setMessages([]);
                      }}
                    >
                      Reset
                    </span>
                  </>
                )}
              </>
            )}
          </p>
        </div>
      )}

      <div className="mt-2 border dark:bg-gray-700 rounded-lg w-[600px] max-w-[80%] max-h-[200px] mx-auto relative mb-3 flex">
        <textarea
          ref={textareaRef}
          className="scrollbar-thin w-[550px] dark:bg-gray-700 px-3 py-4 rounded-lg bg-transparent border-none outline-none text-gray-800 disabled:text-gray-400 dark:text-gray-200 placeholder-gray-500 dark:placeholder-gray-400 disabled:placeholder-gray-200 resize-none disabled:cursor-not-allowed"
          placeholder="Enter the prompt that is really really long..."
          type="text"
          rows={1}
          value={input}
          disabled={status !== "ready"}
          title={status === "ready" ? "Model is ready" : "Model not loaded yet"}
          onKeyDown={(e) => {
            if (
              input.length > 0 &&
              !isRunning &&
              e.key === "Enter" &&
              !e.shiftKey
            ) {
              e.preventDefault(); // Prevent default behavior of Enter key
              onEnter(input);
            }
          }}
          onInput={(e) => setInput(e.target.value)}
        />
        {isRunning ? (
          <div className="cursor-pointer" onClick={onInterrupt}>
            <StopIcon className="h-8 w-8 p-1 rounded-md text-gray-800 dark:text-gray-100 absolute right-3 bottom-3" />
          </div>
        ) : input.length > 0 ? (
          <div className="cursor-pointer" onClick={() => onEnter(input)}>
            <ArrowRightIcon
              className={`h-8 w-8 p-1 bg-gray-800 dark:bg-gray-100 text-white dark:text-black rounded-md absolute right-3 bottom-3`}
            />
          </div>
        ) : (
          <div>
            <ArrowRightIcon
              className={`h-8 w-8 p-1 bg-gray-200 dark:bg-gray-600 text-gray-50 dark:text-gray-800 rounded-md absolute right-3 bottom-3`}
            />
          </div>
        )}
      </div>

      <p className="text-xs text-gray-400 text-center mb-3">
        Open the <b>browser console</b> for live logs.
      </p>
    </div>
  ) : (
    <div className="fixed w-screen h-screen bg-black z-10 bg-opacity-[92%] text-white text-2xl font-semibold flex justify-center items-center text-center">
      WebGPU is not supported
      <br />
      by this browser :&#40;
    </div>
  );
}

export default App;
